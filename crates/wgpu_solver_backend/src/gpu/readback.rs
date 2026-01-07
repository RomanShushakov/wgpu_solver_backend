use bytemuck::{Pod, cast_slice};
use futures::channel::oneshot;
use std::mem::size_of;
use wgpu::PollType;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Device, MapMode, Queue,
};

pub async fn readback_to_vec<T: Pod>(
    device: &Device,
    queue: &Queue,
    src: &Buffer,
    len: usize,
    label: Option<&str>,
) -> Vec<T> {
    let byte_len = (len * size_of::<T>()) as u64;

    // Staging buffer must be MAP_READ + COPY_DST (common wgpu pattern). :contentReference[oaicite:1]{index=1}
    let staging = device.create_buffer(&BufferDescriptor {
        label,
        size: byte_len,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, byte_len);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);

    let (tx, rx) = oneshot::channel();
    slice.map_async(MapMode::Read, move |r| {
        let _ = tx.send(r);
    });

    // Ensure mapping completes.
    device
        .poll(PollType::wait_indefinitely())
        .expect("error at polling");

    rx.await
        .expect("map_async callback dropped")
        .expect("map_async failed");

    let data = slice.get_mapped_range();
    let out = cast_slice::<u8, T>(&data).to_vec();
    drop(data);
    staging.unmap();

    out
}

/// Read a buffer that was created with MAP_READ usage.
/// (No extra staging buffer, no GPU->GPU copy performed here.)
pub async fn read_mapped_buffer_to_vec<T: Pod>(
    device: &Device,
    buffer: &Buffer,
    _len: usize,
) -> Vec<T> {
    let slice = buffer.slice(..);

    let (tx, rx) = oneshot::channel();
    slice.map_async(MapMode::Read, move |r| {
        let _ = tx.send(r);
    });

    device
        .poll(PollType::wait_indefinitely())
        .expect("error at polling");

    rx.await
        .expect("map_async callback dropped")
        .expect("map_async failed");

    let data = slice.get_mapped_range();
    let out = cast_slice::<u8, T>(&data).to_vec();
    drop(data);
    buffer.unmap();

    out
}
