use bytemuck::Pod;
use std::marker::PhantomData;
use std::mem::size_of;
use wgpu::Buffer;

#[derive(Debug)]
pub struct GpuBuffer<T: Pod> {
    pub buffer: Buffer,
    pub len: usize,
    pub _marker: PhantomData<T>,
}

impl<T: Pod> GpuBuffer<T> {
    pub fn byte_len(&self) -> u64 {
        (self.len * size_of::<T>()) as u64
    }
}
