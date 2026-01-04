use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use crate::gpu::context::GpuContext;

/// Pipeline wrapper for `pcg_update_scalars.wgsl`.
///
/// WGSL bindings (group(0)):
///   binding(0) : uniform Params (u32 indices into scalar_results)
///   binding(1) : storage read_write scalar_results (array<f32>)
pub struct PcgUpdateScalarsPipeline {
    pub pipeline: ComputePipeline,
    pub pcg_update_scalars_bind_group_layout: BindGroupLayout,
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

/// Create compute pipeline for `pcg_update_scalars.wgsl`.
pub fn create_pcg_update_scalars_pipeline(ctx: &GpuContext) -> PcgUpdateScalarsPipeline {
    let device = &ctx.device;

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("pcg_update_scalars.wgsl"),
        source: ShaderSource::Wgsl(include_str!("wgsl/pcg_update_scalars.wgsl").into()),
    });

    let pcg_update_scalars_bind_group_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("pcg_update_scalars bgl0"),
            entries: &[
                create_uniform_entry(0),        // Params
                create_storage_entry(1, false), // scalar_results RW
            ],
        });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("pcg_update_scalars pipeline layout"),
        bind_group_layouts: &[&pcg_update_scalars_bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("pcg_update_scalars pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("compute_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    PcgUpdateScalarsPipeline {
        pipeline,
        pcg_update_scalars_bind_group_layout,
    }
}

/// Create bind group for `pcg_update_scalars.wgsl`.
pub fn create_pcg_update_scalars_bind_group(
    device: &Device,
    pcg_update_scalars_bind_group_layout: &BindGroupLayout,
    params_buffer: &Buffer,
    scalar_results_buffer: &Buffer,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("pcg_update_scalars bind group 0"),
        layout: pcg_update_scalars_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: scalar_results_buffer.as_entire_binding(),
            },
        ],
    })
}
