use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use crate::gpu::context::GpuContext;

pub struct SpmvPipeline {
    pub pipeline: ComputePipeline,
    pub spmv_bind_group_layout: BindGroupLayout,
}

fn create_uniform_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_storage_entry(binding: u32, is_read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage {
                read_only: is_read_only,
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn create_spmv_pipeline(ctx: &GpuContext) -> SpmvPipeline {
    let device = &ctx.device;

    // ------------------------------------------------------------------------
    // Shader module
    // ------------------------------------------------------------------------
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("spmv.wgsl"),
        source: ShaderSource::Wgsl(include_str!("wgsl/spmv.wgsl").into()),
    });

    // ------------------------------------------------------------------------
    // Bind group layout (group(0), bindings 0..5), matches spmv.wgsl:
    //   0: Params uniform
    //   1: row_ptr (RO storage)
    //   2: col_idx (RO storage)
    //   3: values  (RO storage)
    //   4: x       (RO storage)
    //   5: y       (RW storage)
    // ------------------------------------------------------------------------
    let spmv_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("spmv bgl0"),
        entries: &[
            create_uniform_entry(0),
            create_storage_entry(1, true),
            create_storage_entry(2, true),
            create_storage_entry(3, true),
            create_storage_entry(4, true),
            create_storage_entry(5, false),
        ],
    });

    // ------------------------------------------------------------------------
    // Pipeline layout
    // (wgpu recent versions use `immediate_size` instead of push_constant_ranges)
    // ------------------------------------------------------------------------
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("spmv pipeline layout"),
        bind_group_layouts: &[&spmv_bind_group_layout],
        immediate_size: 0,
    });

    // ------------------------------------------------------------------------
    // Compute pipeline
    // ------------------------------------------------------------------------
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("spmv pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("compute_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    SpmvPipeline {
        pipeline,
        spmv_bind_group_layout,
    }
}

pub fn create_spmv_bind_group(
    device: &Device,
    spmv_bind_group_layout: &BindGroupLayout,
    params_buffer: &Buffer,
    row_ptr_buffer: &Buffer,
    col_idx_buffer: &Buffer,
    values_buffer: &Buffer,
    x_buffer: &Buffer,
    y_buffer: &Buffer,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("spmv bind group 0"),
        layout: spmv_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: row_ptr_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: col_idx_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: values_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: x_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: y_buffer.as_entire_binding(),
            },
        ],
    })
}
