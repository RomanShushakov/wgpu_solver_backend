use std::cell::Cell;

use bytemuck::{bytes_of, cast_slice};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor};

use crate::{
    compute::{
        dot_partials::{
            DotPartialsPipeline, create_dot_partials_bind_group, create_dot_partials_pipeline,
        },
        dot_reduce::{DotReducePipeline, create_dot_reduce_bind_group, create_dot_reduce_pipeline},
    },
    gpu::{context::GpuContext, readback::read_mapped_buffer_to_vec},
};

pub struct DotScalarExecutor {
    dot_partials_pipeline: DotPartialsPipeline,
    dot_reduce_pipeline: DotReducePipeline,

    // Scratch buffers for reduction ping-pong
    input_buffer: Buffer,
    output_buffer: Buffer,

    max_partials: usize,

    // Scalar outputs (GPU-side, not mappable)
    scalar_results_buffer: Buffer,
    scalar_results_len: usize,

    // Mappable readback buffer (small)
    scalar_readback_buffer: Buffer,

    // Uniform pools
    dot_partials_params_buffers: Vec<Buffer>,
    dot_partials_params_cursor: Cell<usize>,

    dot_reduce_params_buffers: Vec<Buffer>,
    dot_reduce_params_cursor: Cell<usize>,
}

impl DotScalarExecutor {
    pub fn create(ctx: &GpuContext, n_max: usize, scalar_results_len: usize) -> Self {
        let device = &ctx.device;

        let dot_partials_pipeline = create_dot_partials_pipeline(ctx);
        let dot_reduce_pipeline = create_dot_reduce_pipeline(ctx);

        let workgroup_size = 256usize;
        let max_partials = n_max.div_ceil(workgroup_size);

        // Scratch buffers: f32 arrays of length max_partials
        let scratch_bytes = (max_partials * std::mem::size_of::<f32>()) as u64;

        let input_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("dot scratch input"),
            size: scratch_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("dot scratch output"),
            size: scratch_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Scalar results GPU buffer (f32[scalar_results_len])
        let scalar_bytes = (scalar_results_len * std::mem::size_of::<f32>()) as u64;

        let scalar_results_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("scalar results buffer"),
            size: scalar_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Mappable readback buffer
        let scalar_readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("scalar readback buffer"),
            size: scalar_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uniform pools (match fea_app style)
        let pool_size = 16usize;

        let mut dot_partials_params_buffers = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            dot_partials_params_buffers.push(device.create_buffer(&BufferDescriptor {
                label: Some(&format!("dot_partials params {}", i)),
                size: 16, // [n,0,0,0] as u32
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let mut dot_reduce_params_buffers = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            dot_reduce_params_buffers.push(device.create_buffer(&BufferDescriptor {
                label: Some(&format!("dot_reduce params {}", i)),
                size: 16, // [current_len,0,0,0]
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        Self {
            dot_partials_pipeline,
            dot_reduce_pipeline,
            input_buffer,
            output_buffer,
            max_partials,
            scalar_results_buffer,
            scalar_results_len,
            scalar_readback_buffer,
            dot_partials_params_buffers,
            dot_partials_params_cursor: Cell::new(0),
            dot_reduce_params_buffers,
            dot_reduce_params_cursor: Cell::new(0),
        }
    }

    pub fn scalar_results_buffer(&self) -> &Buffer {
        &self.scalar_results_buffer
    }

    fn next_dot_partials_params_buffer(&self) -> &Buffer {
        let i = self.dot_partials_params_cursor.get();
        self.dot_partials_params_cursor
            .set((i + 1) % self.dot_partials_params_buffers.len());
        &self.dot_partials_params_buffers[i]
    }

    fn next_dot_reduce_params_buffer(&self) -> &Buffer {
        let i = self.dot_reduce_params_cursor.get();
        self.dot_reduce_params_cursor
            .set((i + 1) % self.dot_reduce_params_buffers.len());
        &self.dot_reduce_params_buffers[i]
    }

    pub fn reset_params_cursor(&self) {
        self.dot_partials_params_cursor.set(0);
        self.dot_reduce_params_cursor.set(0);
    }

    /// Encode one dot product and store it into scalar_results_buffer[out_index].
    ///
    /// This records multiple compute passes into the provided encoder, but does NOT submit.
    pub fn encode_dot_scalar_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut CommandEncoder,
        a_buffer: &Buffer,
        b_buffer: &Buffer,
        n: u32,
        out_index: u32,
    ) {
        if (out_index as usize) >= self.scalar_results_len {
            panic!("DotScalarExecutor: out_index out of range");
        }

        // n==0: just write 0.0 into the output slot.
        if n == 0 {
            let zero: f32 = 0.0;
            let offset = (out_index as u64) * 4;
            ctx.queue.write_buffer(
                &self.scalar_results_buffer,
                offset,
                bytes_of(&zero),
            );
            return;
        }

        // ---- Pass 1: partial sums into input_buffer ----
        let dot_partials_params = self.next_dot_partials_params_buffer();
        let words: [u32; 4] = [n, 0, 0, 0];
        ctx.queue
            .write_buffer(dot_partials_params, 0, cast_slice(&words));

        let dot_partials_bg = create_dot_partials_bind_group(
            &ctx.device,
            &self.dot_partials_pipeline.dot_partials_bind_group_layout,
            dot_partials_params,
            a_buffer,
            b_buffer,
            &self.input_buffer,
        );

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("dot_partials pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.dot_partials_pipeline.pipeline);
            pass.set_bind_group(0, &dot_partials_bg, &[]);

            let groups = n.div_ceil(256u32);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        // Number of partials produced by pass 1
        let mut current_len: u32 = n.div_ceil(256u32);

        // ---- Pass 2..k: reduce partials until length=1 ----
        let mut current_input = &self.input_buffer;
        let mut current_output = &self.output_buffer;

        while current_len > 1 {
            let reduce_params = self.next_dot_reduce_params_buffer();
            let w: [u32; 4] = [current_len, 0, 0, 0];
            ctx.queue.write_buffer(reduce_params, 0, cast_slice(&w));

            let bg = create_dot_reduce_bind_group(
                &ctx.device,
                &self.dot_reduce_pipeline.dot_reduce_bind_group_layout,
                reduce_params,
                current_input,
                current_output,
            );

            let out_len = current_len.div_ceil(256u32);

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("dot_reduce pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.dot_reduce_pipeline.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(out_len, 1, 1);
            }

            current_len = out_len;
            std::mem::swap(&mut current_input, &mut current_output);
        }

        // ---- Copy final scalar into scalar_results_buffer[out_index] ----
        encoder.copy_buffer_to_buffer(
            current_input, // scalar at offset 0
            0,
            &self.scalar_results_buffer,
            (out_index as u64) * 4,
            4,
        );
    }

    /// Encode one GPU->GPU copy: scalar_readback_buffer <- scalar_results_buffer
    /// Call once per submit after filling all slots you need.
    pub fn encode_copy_scalar_results_to_readback(&self, encoder: &mut CommandEncoder) {
        let bytes = (self.scalar_results_len as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &self.scalar_results_buffer,
            0,
            &self.scalar_readback_buffer,
            0,
            bytes,
        );
    }

    /// After submit, map and read back all scalar slots.
    pub async fn readback_scalar_results(&self, ctx: &GpuContext) -> Vec<f32> {
        read_mapped_buffer_to_vec::<f32>(
            &ctx.device,
            &self.scalar_readback_buffer,
            self.scalar_results_len,
        )
        .await
    }
}
