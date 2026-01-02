use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use crate::gpu::context::GpuContext;

/// Pipeline for the classic AXPY kernel:
///   y[i] = y[i] + alpha * x[i]
/// where `alpha` is provided in a uniform buffer.
pub struct AxpyPipeline {
    pub pipeline: ComputePipeline,
    pub axpy_bind_group_layout: BindGroupLayout,
}

/// Pipeline for AXPY where alpha is read from `scalar_results_buffer[scalar_index]`.
pub struct AxpyFromScalarResultsPipeline {
    pub pipeline: ComputePipeline,
    pub axpy_from_scalar_results_bind_group_layout: BindGroupLayout,
}

/// Pipeline for SCALE where beta is read from `scalar_results_buffer[scalar_index]`:
///   x[i] = x[i] * beta
pub struct ScaleFromScalarResultsPipeline {
    pub pipeline: ComputePipeline,
    pub scale_from_scalar_results_bind_group_layout: BindGroupLayout,
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

pub fn create_axpy_pipeline(ctx: &GpuContext) -> AxpyPipeline {
    let device = &ctx.device;

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("axpy.wgsl"),
        source: ShaderSource::Wgsl(include_str!("wgsl/axpy.wgsl").into()),
    });

    // WGSL group(0) bindings:
    //   binding(0): uniform Params
    //   binding(1): x RO storage
    //   binding(2): y RW storage
    let axpy_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("axpy bgl0"),
        entries: &[
            create_uniform_entry(0),
            create_storage_entry(1, true),
            create_storage_entry(2, false),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("axpy pipeline layout"),
        bind_group_layouts: &[&axpy_bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("axpy pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("compute_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    AxpyPipeline {
        pipeline,
        axpy_bind_group_layout,
    }
}

pub fn create_axpy_from_scalar_results_pipeline(ctx: &GpuContext) -> AxpyFromScalarResultsPipeline {
    let device = &ctx.device;

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("axpy_from_scalar_results.wgsl"),
        source: ShaderSource::Wgsl(include_str!("wgsl/axpy_from_scalar_results.wgsl").into()),
    });

    // WGSL group(0) bindings:
    //   binding(0): uniform Params (n, scalar_index)
    //   binding(1): x RO storage
    //   binding(2): y RW storage
    //   binding(3): scalar_results RO storage
    let axpy_from_scalar_results_bind_group_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("axpy_from_scalar_results bgl0"),
            entries: &[
                create_uniform_entry(0),
                create_storage_entry(1, true),
                create_storage_entry(2, false),
                create_storage_entry(3, true),
            ],
        });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("axpy_from_scalar_results pipeline layout"),
        bind_group_layouts: &[&axpy_from_scalar_results_bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("axpy_from_scalar_results pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("compute_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    AxpyFromScalarResultsPipeline {
        pipeline,
        axpy_from_scalar_results_bind_group_layout,
    }
}

pub fn create_scale_from_scalar_results_pipeline(
    ctx: &GpuContext,
) -> ScaleFromScalarResultsPipeline {
    let device = &ctx.device;

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("scale_from_scalar_results.wgsl"),
        source: ShaderSource::Wgsl(include_str!("wgsl/scale_from_scalar_results.wgsl").into()),
    });

    // WGSL group(0) bindings:
    //   binding(0): uniform Params (n, scalar_index)
    //   binding(1): x RW storage
    //   binding(2): scalar_results RO storage
    let scale_from_scalar_results_bind_group_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("scale_from_scalar_results bgl0"),
            entries: &[
                create_uniform_entry(0),
                create_storage_entry(1, false),
                create_storage_entry(2, true),
            ],
        });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("scale_from_scalar_results pipeline layout"),
        bind_group_layouts: &[&scale_from_scalar_results_bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("scale_from_scalar_results pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("compute_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    ScaleFromScalarResultsPipeline {
        pipeline,
        scale_from_scalar_results_bind_group_layout,
    }
}

pub fn create_axpy_bind_group(
    device: &Device,
    axpy_bind_group_layout: &BindGroupLayout,
    params_buffer: &Buffer, // binding(0)
    x_buffer: &Buffer,      // binding(1)
    y_buffer: &Buffer,      // binding(2)
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("axpy bind group 0"),
        layout: axpy_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: x_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: y_buffer.as_entire_binding(),
            },
        ],
    })
}

pub fn create_axpy_from_scalar_results_bind_group(
    device: &Device,
    axpy_from_scalar_results_bind_group_layout: &BindGroupLayout,
    params_buffer: &Buffer,         // binding(0)
    x_buffer: &Buffer,              // binding(1)
    y_buffer: &Buffer,              // binding(2)
    scalar_results_buffer: &Buffer, // binding(3)
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("axpy_from_scalar_results bind group 0"),
        layout: axpy_from_scalar_results_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: x_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: y_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: scalar_results_buffer.as_entire_binding(),
            },
        ],
    })
}

pub fn create_scale_from_scalar_results_bind_group(
    device: &Device,
    scale_from_scalar_results_bind_group_layout: &BindGroupLayout,
    params_buffer: &Buffer,         // binding(0)
    x_buffer: &Buffer,              // binding(1)
    scalar_results_buffer: &Buffer, // binding(2)
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("scale_from_scalar_results bind group 0"),
        layout: scale_from_scalar_results_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: x_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: scalar_results_buffer.as_entire_binding(),
            },
        ],
    })
}
