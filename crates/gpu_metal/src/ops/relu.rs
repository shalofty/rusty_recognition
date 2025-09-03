use crate::{MetalDevice, MetalBuffer, LaunchConfig, dispatch_kernel};

pub fn relu_forward_gpu(
    device: &mut MetalDevice,
    input: &MetalBuffer,
    output: &mut MetalBuffer,
    n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = device.get_or_create_pipeline("relu_forward")?.clone();
    let command_buffer = device.new_command_buffer();
    
    let encoder = command_buffer.new_compute_command_encoder();
    
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&output.buffer), 0);
    encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &(n as u32) as *const u32 as *const _);
    
    let config = LaunchConfig::new_1d(n, 256);
    dispatch_kernel(&encoder, &pipeline, &config);
    
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    
    Ok(())
}

pub fn relu_backward_gpu(
    device: &mut MetalDevice,
    grad_output: &MetalBuffer,
    input: &MetalBuffer,
    grad_input: &mut MetalBuffer,
    n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = device.get_or_create_pipeline("drelu")?.clone();
    let command_buffer = device.new_command_buffer();
    
    let encoder = command_buffer.new_compute_command_encoder();
    
    encoder.set_buffer(0, Some(&grad_output.buffer), 0);
    encoder.set_buffer(1, Some(&input.buffer), 0);
    encoder.set_buffer(2, Some(&grad_input.buffer), 0);
    encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(n as u32) as *const u32 as *const _);
    
    let config = LaunchConfig::new_1d(n, 256);
    dispatch_kernel(&encoder, &pipeline, &config);
    
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    
    Ok(())
}