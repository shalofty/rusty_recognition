use metal::*;

pub struct LaunchConfig {
    pub grid_size: MTLSize,
    pub block_size: MTLSize,
}

impl LaunchConfig {
    pub fn new_1d(total_threads: usize, threads_per_group: usize) -> Self {
        let groups = (total_threads + threads_per_group - 1) / threads_per_group;
        
        Self {
            grid_size: MTLSize::new(groups as u64, 1, 1),
            block_size: MTLSize::new(threads_per_group as u64, 1, 1),
        }
    }
    
    pub fn new_2d(width: usize, height: usize, threads_per_group_x: usize, threads_per_group_y: usize) -> Self {
        let groups_x = (width + threads_per_group_x - 1) / threads_per_group_x;
        let groups_y = (height + threads_per_group_y - 1) / threads_per_group_y;
        
        Self {
            grid_size: MTLSize::new(groups_x as u64, groups_y as u64, 1),
            block_size: MTLSize::new(threads_per_group_x as u64, threads_per_group_y as u64, 1),
        }
    }
    
    pub fn new_3d(width: usize, height: usize, depth: usize, 
                  threads_per_group_x: usize, threads_per_group_y: usize, threads_per_group_z: usize) -> Self {
        let groups_x = (width + threads_per_group_x - 1) / threads_per_group_x;
        let groups_y = (height + threads_per_group_y - 1) / threads_per_group_y;
        let groups_z = (depth + threads_per_group_z - 1) / threads_per_group_z;
        
        Self {
            grid_size: MTLSize::new(groups_x as u64, groups_y as u64, groups_z as u64),
            block_size: MTLSize::new(threads_per_group_x as u64, threads_per_group_y as u64, threads_per_group_z as u64),
        }
    }
}

pub fn dispatch_kernel(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    config: &LaunchConfig
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.dispatch_thread_groups(config.grid_size, config.block_size);
}