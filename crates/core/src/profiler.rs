use std::collections::HashMap;
use std::time::{Instant, Duration};

// M6 Improvement: Performance monitoring and profiling capabilities
pub struct PerformanceProfiler {
    kernel_times: HashMap<String, Vec<f64>>,
    memory_usage: Vec<usize>,
    gpu_utilization: Vec<f32>,
    epoch_times: Vec<f64>,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            kernel_times: HashMap::new(),
            memory_usage: Vec::new(),
            gpu_utilization: Vec::new(),
            epoch_times: Vec::new(),
        }
    }
    
    pub fn profile_kernel<F, R>(&mut self, name: &str, op: F) -> anyhow::Result<R>
    where F: FnOnce() -> anyhow::Result<R>
    {
        let start = Instant::now();
        let result = op()?;
        let duration = start.elapsed().as_secs_f64() * 1000.0; // Convert to ms
        
        self.kernel_times.entry(name.to_string()).or_default().push(duration);
        Ok(result)
    }
    
    pub fn record_memory_usage(&mut self, bytes: usize) {
        self.memory_usage.push(bytes);
    }
    
    pub fn record_gpu_utilization(&mut self, utilization: f32) {
        self.gpu_utilization.push(utilization);
    }
    
    pub fn record_epoch_time(&mut self, duration: Duration) {
        self.epoch_times.push(duration.as_secs_f64());
    }
    
    pub fn record_kernel_time(&mut self, name: &str, duration_ms: f64) {
        self.kernel_times.entry(name.to_string()).or_default().push(duration_ms);
    }
    
    pub fn get_kernel_stats(&self, name: &str) -> Option<KernelStats> {
        self.kernel_times.get(name).map(|times| {
            let sum: f64 = times.iter().sum();
            let count = times.len();
            let mean = sum / count as f64;
            
            let variance = times.iter()
                .map(|time| (time - mean).powi(2))
                .sum::<f64>() / count as f64;
            let std_dev = variance.sqrt();
            
            let min = times.iter().copied().fold(f64::INFINITY, f64::min);
            let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            
            KernelStats {
                name: name.to_string(),
                call_count: count,
                total_time: sum,
                avg_time: mean,
                min_time: min,
                max_time: max,
                std_dev,
            }
        })
    }
    
    pub fn get_throughput_stats(&self) -> ThroughputStats {
        let total_epochs = self.epoch_times.len();
        let total_time: f64 = self.epoch_times.iter().sum();
        let avg_epoch_time = if total_epochs > 0 {
            total_time / total_epochs as f64
        } else {
            0.0
        };
        
        let peak_memory = self.memory_usage.iter().copied().max().unwrap_or(0);
        let avg_gpu_util = if !self.gpu_utilization.is_empty() {
            self.gpu_utilization.iter().sum::<f32>() / self.gpu_utilization.len() as f32
        } else {
            0.0
        };
        
        ThroughputStats {
            total_epochs,
            total_training_time: total_time,
            avg_epoch_time,
            peak_memory_usage: peak_memory,
            avg_gpu_utilization: avg_gpu_util,
        }
    }
    
    pub fn print_summary(&self) {
        println!("=== Performance Profile Summary ===");
        
        // Kernel timing stats
        println!("\nKernel Performance:");
        let mut kernel_names: Vec<_> = self.kernel_times.keys().collect();
        kernel_names.sort();
        
        for name in kernel_names {
            if let Some(stats) = self.get_kernel_stats(name) {
                println!("  {}: {} calls, {:.2}ms avg, {:.2}ms total", 
                    stats.name, stats.call_count, stats.avg_time, stats.total_time);
            }
        }
        
        // Throughput stats
        let throughput = self.get_throughput_stats();
        println!("\nThroughput Stats:");
        println!("  Epochs: {}", throughput.total_epochs);
        println!("  Total training time: {:.2}s", throughput.total_training_time);
        println!("  Average epoch time: {:.2}s", throughput.avg_epoch_time);
        println!("  Peak memory usage: {:.2} MB", throughput.peak_memory_usage as f64 / 1024.0 / 1024.0);
        println!("  Average GPU utilization: {:.1}%", throughput.avg_gpu_utilization * 100.0);
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct KernelStats {
    pub name: String,
    pub call_count: usize,
    pub total_time: f64,    // milliseconds
    pub avg_time: f64,      // milliseconds
    pub min_time: f64,      // milliseconds
    pub max_time: f64,      // milliseconds
    pub std_dev: f64,       // milliseconds
}

#[derive(Debug, Clone)]
pub struct ThroughputStats {
    pub total_epochs: usize,
    pub total_training_time: f64,    // seconds
    pub avg_epoch_time: f64,         // seconds
    pub peak_memory_usage: usize,    // bytes
    pub avg_gpu_utilization: f32,    // 0.0 to 1.0
}

// Memory pool for optimized tensor allocation
pub struct TensorMemoryPool<B> {
    free_buffers: HashMap<usize, Vec<B>>, // size -> list of free buffers
    allocated_count: usize,
    peak_usage: usize,
}

impl<B> TensorMemoryPool<B> {
    pub fn new() -> Self {
        Self {
            free_buffers: HashMap::new(),
            allocated_count: 0,
            peak_usage: 0,
        }
    }
    
    pub fn get_buffer(&mut self, size: usize) -> Option<B> {
        if let Some(buffers) = self.free_buffers.get_mut(&size) {
            buffers.pop()
        } else {
            None
        }
    }
    
    pub fn return_buffer(&mut self, size: usize, buffer: B) {
        self.free_buffers.entry(size).or_default().push(buffer);
    }
    
    pub fn get_stats(&self) -> MemoryPoolStats {
        let total_cached: usize = self.free_buffers.values()
            .map(|buffers| buffers.len())
            .sum();
        
        MemoryPoolStats {
            allocated_count: self.allocated_count,
            cached_count: total_cached,
            peak_usage: self.peak_usage,
        }
    }
}

impl<B> Default for TensorMemoryPool<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub allocated_count: usize,
    pub cached_count: usize,
    pub peak_usage: usize,
}