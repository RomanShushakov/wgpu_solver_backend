use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor,
};

use crate::compute::spmv::{SpmvPipeline, create_spmv_bind_group, create_spmv_pipeline};
use crate::gpu::context::GpuContext;

/// SpmvExecutor (native wgpu version)
///
/// Owns persistent GPU resources for CSR SpMV:
///   - CSR structure buffers: row_ptr, col_idx, values (uploaded once)
///   - params uniform buffer: n_rows (written once)
///   - internal x/y vectors used by the shader:
///       x_buffer: input vector for A*x (copy your current vector into it)
///       y_buffer: output vector (SpMV result), reused every call
pub struct SpmvExecutor {
    n_rows: u32,

    // Pipeline + bind group
    spmv_pipeline: SpmvPipeline,
    spmv_bind_group: BindGroup,

    // Persistent buffers (created once)
    #[allow(dead_code)]
    params_buffer: Buffer,
    #[allow(dead_code)]
    row_ptr_buffer: Buffer,
    #[allow(dead_code)]
    col_idx_buffer: Buffer,
    #[allow(dead_code)]
    values_buffer: Buffer,
    x_buffer: Buffer,
    y_buffer: Buffer,
}

impl SpmvExecutor {
    pub fn create(
        ctx: &GpuContext,
        n_rows: u32,
        row_ptr_u32: &[u32],
        col_idx_u32: &[u32],
        values_f32: &[f32],
    ) -> Self {
        let device = &ctx.device;

        // 1) Create pipeline (once).
        let spmv_pipeline = create_spmv_pipeline(ctx);

        // 2) Params uniform (once): { n_rows, 0, 0, 0 }
        let params_words: [u32; 4] = [n_rows, 0, 0, 0];
        let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("spmv params"),
            contents: bytemuck::cast_slice(&params_words),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // 3) CSR buffers (once).
        // row_ptr length must be n_rows + 1; col_idx/values length must be nnz.
        let row_ptr_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("spmv row_ptr"),
            contents: bytemuck::cast_slice(row_ptr_u32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let col_idx_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("spmv col_idx"),
            contents: bytemuck::cast_slice(col_idx_u32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let values_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("spmv values"),
            contents: bytemuck::cast_slice(values_f32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        // 4) Internal x/y buffers (reused every encode_spmv call).
        // x is read-only in WGSL; needs COPY_DST (we fill via GPU->GPU copy).
        let x_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("spmv x"),
            size: (n_rows as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // y is read_write and used by other kernels; keep COPY_SRC for optional debug readback.
        let y_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("spmv y"),
            size: (n_rows as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 5) Bind group (once).
        let spmv_bind_group = create_spmv_bind_group(
            device,
            &spmv_pipeline.spmv_bind_group_layout,
            &params_buffer,
            &row_ptr_buffer,
            &col_idx_buffer,
            &values_buffer,
            &x_buffer,
            &y_buffer,
        );

        Self {
            n_rows,
            spmv_pipeline,
            spmv_bind_group,
            params_buffer,
            row_ptr_buffer,
            col_idx_buffer,
            values_buffer,
            x_buffer,
            y_buffer,
        }
    }

    /// Encode the CSR SpMV compute pass:
    ///   y_buffer = A * x_buffer
    pub fn encode_spmv(&self, encoder: &mut CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("spmv pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.spmv_pipeline.pipeline);
        pass.set_bind_group(0, &self.spmv_bind_group, &[]);

        let workgroup_size = 256u32;
        let groups_x = (self.n_rows + workgroup_size - 1) / workgroup_size;
        pass.dispatch_workgroups(groups_x, 1, 1);
    }

    /// Output buffer produced by encode_spmv(): y = A*x
    pub fn y_buffer(&self) -> &Buffer {
        &self.y_buffer
    }

    /// Encode a GPU->GPU copy into the executor's internal x_buffer:
    ///   x_buffer <- src_gpu
    ///
    /// `n_bytes` should be `n_rows * sizeof(f32)`.
    pub fn encode_copy_x_from(&self, encoder: &mut CommandEncoder, src_gpu: &Buffer, n_bytes: u64) {
        encoder.copy_buffer_to_buffer(src_gpu, 0, &self.x_buffer, 0, n_bytes);
    }
}
