use metal::*;
use std::collections::HashMap;
use anyhow::{Result, Context};

pub struct MetalDevice {
    pub device: Device,
    pub command_queue: CommandQueue,
    pub library: Library,
    pipeline_cache: HashMap<String, ComputePipelineState>,
}

impl MetalDevice {
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .context("No Metal device found")?;
        
        let command_queue = device.new_command_queue();
        
        let library = device.new_library_with_source(include_str!("../shaders/kernels.metal"), &CompileOptions::new())
            .map_err(|e| anyhow::anyhow!("Failed to compile Metal shaders: {}", e))?;
        
        Ok(Self {
            device,
            command_queue,
            library,
            pipeline_cache: HashMap::new(),
        })
    }
    
    pub fn get_or_create_pipeline(&mut self, function_name: &str) -> Result<&ComputePipelineState> {
        if !self.pipeline_cache.contains_key(function_name) {
            let function = self.library.get_function(function_name, None)
                .map_err(|e| anyhow::anyhow!("Function '{}' not found in library: {}", function_name, e))?;
            
            let pipeline = self.device.new_compute_pipeline_state_with_function(&function)
                .map_err(|e| anyhow::anyhow!("Failed to create pipeline for '{}': {}", function_name, e))?;
            
            self.pipeline_cache.insert(function_name.to_string(), pipeline);
        }
        
        Ok(self.pipeline_cache.get(function_name).unwrap())
    }
    
    pub fn new_command_buffer(&self) -> CommandBuffer {
        self.command_queue.new_command_buffer().to_owned()
    }
}