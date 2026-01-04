use bytemuck::Pod;
use std::marker::PhantomData;
use std::mem::size_of;
use thiserror::Error;
use wgpu::{
    Adapter, Backend, Backends, BufferDescriptor, BufferUsages, Device, DeviceDescriptor,
    DeviceType, ExperimentalFeatures, Features, Instance, InstanceDescriptor, Limits, MemoryHints,
    PowerPreference, Queue, RequestAdapterOptions, Trace,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::gpu::{buffer::GpuBuffer, readback::readback_to_vec};

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter found")]
    NoAdapter,
    #[error("request device failed: {0}")]
    RequestDevice(String),
}

#[derive(Debug, Clone, Copy)]
pub enum GpuBackend {
    Auto,
    Vulkan,
    Dx12,
    Metal,
}

#[derive(Debug)]
pub struct AdapterInfo {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: DeviceType,
    pub backend: Backend,
}

#[derive(Debug)]
pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub adapter_info: AdapterInfo,
}

fn backend_bits(gpu_backend: GpuBackend) -> Backends {
    match gpu_backend {
        GpuBackend::Auto => Backends::all(),
        GpuBackend::Vulkan => Backends::VULKAN,
        GpuBackend::Dx12 => Backends::DX12,
        GpuBackend::Metal => Backends::METAL,
    }
}

impl GpuContext {
    /// Minimal headless compute context.
    /// Explicit and boring by design.
    pub async fn create(gpu_backend: GpuBackend) -> Result<Self, GpuError> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: backend_bits(gpu_backend),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .or(Err(GpuError::NoAdapter))?;

        let info = adapter.get_info();
        let adapter_info = AdapterInfo {
            name: info.name,
            vendor: info.vendor,
            device: info.device,
            device_type: info.device_type,
            backend: info.backend,
        };

        let required_features = Features::empty();
        let required_limits = Limits::default();
        let experimental_features = ExperimentalFeatures::disabled();
        let memory_hints = MemoryHints::default();
        let trace = Trace::default();

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("wgpu_solver_backend_device"),
                required_features,
                required_limits,
                experimental_features,
                memory_hints,
                trace,
            })
            .await
            .map_err(|e| GpuError::RequestDevice(format!("{e:?}")))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            adapter_info,
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}, backend={:?}, vendor=0x{:04x}, device=0x{:04x})",
            self.adapter_info.name,
            self.adapter_info.device_type,
            self.adapter_info.backend,
            self.adapter_info.vendor,
            self.adapter_info.device
        )
    }

    pub fn create_storage_buffer<T: Pod>(
        &self,
        label: &str,
        data: &[T],
        extra_usage: BufferUsages,
    ) -> GpuBuffer<T> {
        let usage =
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST | extra_usage;

        let buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        });

        GpuBuffer {
            buffer,
            len: data.len(),
            _marker: PhantomData,
        }
    }

    pub fn create_storage_buffer_uninit<T: Pod>(
        &self,
        label: &str,
        len: usize,
        extra_usage: BufferUsages,
    ) -> GpuBuffer<T> {
        let byte_len = (len * size_of::<T>()) as u64;
        let usage =
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST | extra_usage;

        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: byte_len,
            usage,
            mapped_at_creation: false,
        });

        GpuBuffer {
            buffer,
            len,
            _marker: PhantomData,
        }
    }

    pub async fn readback<T: Pod>(&self, buf: &GpuBuffer<T>) -> Vec<T> {
        readback_to_vec::<T>(
            &self.device,
            &self.queue,
            &buf.buffer,
            buf.len,
            Some("readback_staging"),
        )
        .await
    }
}
