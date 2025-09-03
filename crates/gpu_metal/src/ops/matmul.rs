use crate::{MetalDevice, MetalBuffer, LaunchConfig, dispatch_kernel};

pub fn matmul_gpu(
    device: &mut MetalDevice,
    a: &MetalBuffer,
    b: &MetalBuffer,
    c: &mut MetalBuffer,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = device.get_or_create_pipeline("naive_gemm")?.clone();
    let command_buffer = device.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    
    encoder.set_buffer(0, Some(&a.buffer), 0);
    encoder.set_buffer(1, Some(&b.buffer), 0);
    encoder.set_buffer(2, Some(&c.buffer), 0);
    
    let params = [m as u32, n as u32, k as u32];
    encoder.set_bytes(3, std::mem::size_of_val(&params) as u64, params.as_ptr() as *const _);
    encoder.set_bytes(4, std::mem::size_of::<f32>() as u64, &alpha as *const f32 as *const _);
    encoder.set_bytes(5, std::mem::size_of::<f32>() as u64, &beta as *const f32 as *const _);
    
    let config = LaunchConfig::new_2d(m, n, 16, 16);
    dispatch_kernel(&encoder, &pipeline, &config);
    
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    
    Ok(())
}

pub fn matmul_tiled_gpu(
    device: &mut MetalDevice,
    a: &MetalBuffer,
    b: &MetalBuffer,
    c: &mut MetalBuffer,
    m: usize,
    n: usize,
    k: usize,
    tile_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = device.get_or_create_pipeline("matmul_tiled")?.clone();
    
    let command_buffer = device.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    
    encoder.set_buffer(0, Some(&a.buffer), 0);
    encoder.set_buffer(1, Some(&b.buffer), 0);
    encoder.set_buffer(2, Some(&c.buffer), 0);
    
    let params = [m as u32, n as u32, k as u32];
    encoder.set_bytes(3, std::mem::size_of_val(&params) as u64, params.as_ptr() as *const _);
    
    let config = LaunchConfig::new_2d(
        (m + tile_size - 1) / tile_size,
        (n + tile_size - 1) / tile_size,
        tile_size,
        tile_size,
    );
    dispatch_kernel(&encoder, &pipeline, &config);
    
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    
    Ok(())
}