use wgpu::{
    Buffer, BufferUsages, CommandEncoder, Device,
    util::{BufferInitDescriptor, DeviceExt},
};

/// Encode a tiny CPU->GPU write into a *storage* buffer at element index `slot`.
/// This is the native wgpu equivalent of your WebGPU "staging write into storage buffer index".
///
/// Assumptions:
/// - buffer contains f32 elements (so offset = slot * 4).
/// - buffer has COPY_DST usage.
/// - we keep this explicit and tiny (one 4-byte staging buffer).
pub fn encode_write_f32_into_storage_buffer_at_index(
    device: &Device,
    encoder: &mut CommandEncoder,
    dst_storage_buffer: &Buffer,
    slot: u32,
    value: f32,
    label: &str,
) {
    let staging = device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(&value),
        usage: BufferUsages::COPY_SRC,
    });

    encoder.copy_buffer_to_buffer(&staging, 0, dst_storage_buffer, (slot as u64) * 4, 4);
}
