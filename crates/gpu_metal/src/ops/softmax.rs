use crate::{MetalDevice, MetalBuffer, LaunchConfig, dispatch_kernel};

pub fn softmax_forward_gpu(
    device: &mut MetalDevice,
    logits: &MetalBuffer,
    output: &mut MetalBuffer,
    batch_size: usize,
    num_classes: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = device.get_or_create_pipeline("softmax_forward")?.clone();
    
    let command_buffer = device.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    
    encoder.set_buffer(0, Some(&logits.buffer), 0);
    encoder.set_buffer(1, Some(&output.buffer), 0);
    encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &(batch_size as u32) as *const u32 as *const _);
    encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(num_classes as u32) as *const u32 as *const _);
    
    let config = LaunchConfig::new_1d(batch_size, 1);
    dispatch_kernel(&encoder, &pipeline, &config);
    
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    
    Ok(())
}

pub fn softmax_cross_entropy_gpu(
    device: &mut MetalDevice,
    logits: &MetalBuffer,
    labels: &MetalBuffer,
    loss: &mut MetalBuffer,
    grad_logits: &mut MetalBuffer,
    batch_size: usize,
    num_classes: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = device.get_or_create_pipeline("softmax_xent")?.clone();
    
    let command_buffer = device.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    
    encoder.set_buffer(0, Some(&logits.buffer), 0);
    encoder.set_buffer(1, Some(&labels.buffer), 0);
    encoder.set_buffer(2, Some(&loss.buffer), 0);
    encoder.set_buffer(3, Some(&grad_logits.buffer), 0);
    encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(batch_size as u32) as *const u32 as *const _);
    encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &(num_classes as u32) as *const u32 as *const _);
    
    let config = LaunchConfig::new_1d(batch_size, 1);
    dispatch_kernel(&encoder, &pipeline, &config);
    
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    
    Ok(())
}