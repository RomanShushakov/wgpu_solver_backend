use std::cell::Cell;

use bytemuck::cast_slice;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor};

use crate::compute::pcg_update_scalars::{
    PcgUpdateScalarsPipeline, create_pcg_update_scalars_bind_group,
    create_pcg_update_scalars_pipeline,
};
use crate::gpu::context::GpuContext;

/// Encodes the tiny "scalar update" pass used by PCG.
///
/// Reads dot-products from `scalar_results_buffer` and writes:
///   alpha       = rz_old / p_ap
///   minus_alpha = -alpha
///   beta        = rz_new / rz_old
///
/// Params uniform layout (8 u32 = 32 bytes), matches WGSL:
///   [p_ap_index, rz_new_index, rz_old_index,
///    alpha_index, minus_alpha_index, beta_index,
///    0, 0]
///
/// IMPORTANT: we keep a small pool of params buffers so multiple passes can be
/// encoded per command buffer safely.
pub struct PcgUpdateScalarsExecutor {
    pcg_update_scalars_pipeline: PcgUpdateScalarsPipeline,

    params_buffers: Vec<Buffer>,
    params_buffers_cursor: Cell<usize>,
}

impl PcgUpdateScalarsExecutor {
    pub fn create(ctx: &GpuContext) -> Self {
        let device = &ctx.device;

        let pcg_update_scalars_pipeline = create_pcg_update_scalars_pipeline(ctx);

        // Same logic as in fea_app: 2 uses per iteration; 4 is safe headroom.
        let params_buffers_pool_size = 4usize;

        let mut params_buffers = Vec::with_capacity(params_buffers_pool_size);
        for i in 0..params_buffers_pool_size {
            let buf = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("pcg update scalars params {}", i)),
                size: 32, // 8 u32
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            params_buffers.push(buf);
        }

        Self {
            pcg_update_scalars_pipeline,
            params_buffers,
            params_buffers_cursor: Cell::new(0),
        }
    }

    pub fn reset_params_cursor(&self) {
        self.params_buffers_cursor.set(0);
    }

    fn next_params_buffer(&self) -> &Buffer {
        let i = self.params_buffers_cursor.get();
        self.params_buffers_cursor
            .set((i + 1) % self.params_buffers.len());
        &self.params_buffers[i]
    }

    /// Encode one `pcg_update_scalars.wgsl` dispatch.
    pub fn encode_update_scalars(
        &self,
        ctx: &GpuContext,
        encoder: &mut CommandEncoder,
        scalar_results_buffer: &Buffer,
        p_ap_index: u32,
        rz_new_index: u32,
        rz_old_index: u32,
        alpha_index: u32,
        minus_alpha_index: u32,
        beta_index: u32,
    ) {
        // Allocate params from pool and upload indices.
        let params_buffer = self.next_params_buffer();

        let words: [u32; 8] = [
            p_ap_index,        // offset 0
            rz_new_index,      // offset 4
            rz_old_index,      // offset 8
            alpha_index,       // offset 12
            minus_alpha_index, // offset 16
            beta_index,        // offset 20
            0,                 // pad
            0,                 // pad
        ];

        ctx.queue.write_buffer(params_buffer, 0, cast_slice(&words));

        // Bind params + scalar_results.
        let bind_group = create_pcg_update_scalars_bind_group(
            &ctx.device,
            &self
                .pcg_update_scalars_pipeline
                .pcg_update_scalars_bind_group_layout,
            params_buffer,
            scalar_results_buffer,
        );

        // Dispatch 1 workgroup (WGSL uses @workgroup_size(1)).
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("pcg_update_scalars pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pcg_update_scalars_pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}
