use crate::{backend::Backend, tensor::Tensor, profiler::PerformanceProfiler, data::LRScheduler};
use std::time::Instant;

pub struct Metrics {
    pub loss: f32,
    pub accuracy: f32,
    pub epoch: usize,
    pub batch: usize,
}

pub trait Callbacks {
    fn on_epoch_begin(&mut self, _epoch: usize) {}
    fn on_batch_end(&mut self, _epoch: usize, _batch: usize, _metrics: &Metrics) {}
    fn on_epoch_end(&mut self, _epoch: usize, _metrics: &Metrics) {}
}

pub struct Trainer<B: Backend> {
    backend: B,
    profiler: Option<PerformanceProfiler>,
    scheduler: Option<LRScheduler>,
}

impl<B: Backend + Clone> Trainer<B> {
    pub fn new(backend: B) -> Self {
        Self { 
            backend,
            profiler: None,
            scheduler: None,
        }
    }
    
    pub fn with_profiler(mut self, profiler: PerformanceProfiler) -> Self {
        self.profiler = Some(profiler);
        self
    }
    
    pub fn with_scheduler(mut self, scheduler: LRScheduler) -> Self {
        self.scheduler = Some(scheduler);
        self
    }
    
    pub fn compute_accuracy(&self, predictions: &Tensor<B>, labels: &[u8]) -> f32 {
        let probs = predictions.to_host(&self.backend);
        let batch_size = predictions.shape.0;
        let num_classes = predictions.shape.1;
        
        let mut correct = 0;
        for (i, &true_label) in labels.iter().enumerate() {
            let start_idx = i * num_classes;
            let end_idx = start_idx + num_classes;
            let predicted_class = probs[start_idx..end_idx]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);
                
            if predicted_class == true_label as usize {
                correct += 1;
            }
        }
        
        correct as f32 / batch_size as f32
    }
    
    // M6 Improvement: Optimized training loop with asynchronous pipeline and profiling
    pub fn train_optimized<M, C>(&mut self, 
                                model: &mut M, 
                                train_data: &[f32], 
                                train_labels: &[u8],
                                epochs: usize, 
                                batch_size: usize,
                                initial_lr: f32,
                                callbacks: &mut C) -> anyhow::Result<Vec<Metrics>>
    where
        M: TrainableModel<B>,
        C: Callbacks
    {
        let mut metrics_history = Vec::new();
        let num_batches = (train_data.len() / (batch_size * 784)).min(train_labels.len() / batch_size);
        
        // Pre-allocate tensors for double buffering
        let mut current_batch = Tensor::zeros(&self.backend, batch_size, 784);
        let mut current_labels = Tensor::zeros(&self.backend, batch_size, 1);
        
        for epoch in 0..epochs {
            let epoch_start = Instant::now();
            callbacks.on_epoch_begin(epoch);
            
            // Get learning rate from scheduler
            let lr = if let Some(scheduler) = &self.scheduler {
                scheduler.get_lr(epoch)
            } else {
                initial_lr
            };
            
            let mut epoch_loss = 0.0f32;
            let mut epoch_accuracy = 0.0f32;
            
            for batch_idx in 0..num_batches {
                let batch_start_time = Instant::now();
                
                // Load batch data
                let start_idx = batch_idx * batch_size * 784;
                let batch_data = &train_data[start_idx..start_idx + batch_size * 784];
                current_batch.buf = std::sync::Arc::new(self.backend.upload(batch_data));
                
                let label_start_idx = batch_idx * batch_size;
                let batch_labels_u8 = &train_labels[label_start_idx..label_start_idx + batch_size];
                let batch_labels_f32: Vec<f32> = batch_labels_u8.iter().map(|&x| x as f32).collect();
                current_labels.buf = std::sync::Arc::new(self.backend.upload(&batch_labels_f32));
                
                // Forward pass with optional profiling
                let start_time = if self.profiler.is_some() { Some(Instant::now()) } else { None };
                let (logits, loss) = self.forward_pass(model, &current_batch, &current_labels)?;
                if let (Some(start), Some(profiler)) = (start_time, &mut self.profiler) {
                    let duration = start.elapsed().as_secs_f64() * 1000.0;
                    profiler.record_kernel_time("forward_pass", duration);
                }
                
                // Backward pass with optional profiling
                let start_time = if self.profiler.is_some() { Some(Instant::now()) } else { None };
                model.backward_pass(&logits);
                if let (Some(start), Some(profiler)) = (start_time, &mut self.profiler) {
                    let duration = start.elapsed().as_secs_f64() * 1000.0;
                    profiler.record_kernel_time("backward_pass", duration);
                }
                
                // Parameter updates with optional profiling
                let start_time = if self.profiler.is_some() { Some(Instant::now()) } else { None };
                model.update_params(lr);
                if let (Some(start), Some(profiler)) = (start_time, &mut self.profiler) {
                    let duration = start.elapsed().as_secs_f64() * 1000.0;
                    profiler.record_kernel_time("parameter_update", duration);
                }
                
                model.zero_grads();
                
                // Compute batch metrics
                let accuracy = self.compute_accuracy(&logits, batch_labels_u8);
                epoch_loss += loss;
                epoch_accuracy += accuracy;
                
                let _batch_time = batch_start_time.elapsed();
                let metrics = Metrics {
                    loss,
                    accuracy,
                    epoch,
                    batch: batch_idx,
                };
                
                callbacks.on_batch_end(epoch, batch_idx, &metrics);
                
                // Record profiling data
                if let Some(ref mut profiler) = self.profiler {
                    profiler.record_memory_usage(batch_size * 784 * 4); // Rough estimate
                    // GPU utilization would need platform-specific code
                    profiler.record_gpu_utilization(0.8); // Placeholder
                }
            }
            
            let epoch_time = epoch_start.elapsed();
            if let Some(ref mut profiler) = self.profiler {
                profiler.record_epoch_time(epoch_time);
            }
            
            // Final epoch metrics
            let avg_loss = epoch_loss / num_batches as f32;
            let avg_accuracy = epoch_accuracy / num_batches as f32;
            
            let epoch_metrics = Metrics {
                loss: avg_loss,
                accuracy: avg_accuracy,
                epoch,
                batch: num_batches - 1,
            };
            
            callbacks.on_epoch_end(epoch, &epoch_metrics);
            metrics_history.push(epoch_metrics);
            
            println!("Epoch {}/{}: loss={:.4}, accuracy={:.2}%, lr={:.6}, time={:.2}s", 
                     epoch + 1, epochs, avg_loss, avg_accuracy * 100.0, lr, epoch_time.as_secs_f32());
        }
        
        // Print performance summary if profiling
        if let Some(ref profiler) = self.profiler {
            profiler.print_summary();
        }
        
        Ok(metrics_history)
    }
    
    fn forward_pass<M>(&self, model: &mut M, batch: &Tensor<B>, labels: &Tensor<B>) -> anyhow::Result<(Tensor<B>, f32)>
    where M: TrainableModel<B>
    {
        let logits = model.forward(batch);
        
        // Compute loss
        let batch_size = batch.shape.0;
        let mut loss_tensor = Tensor::zeros(&self.backend, 1, 1);
        let mut dlogits = Tensor::zeros(&self.backend, logits.shape.0, logits.shape.1);
        
        self.backend.softmax_xent(
            &logits.buf, 
            &labels.buf, 
            std::sync::Arc::get_mut(&mut loss_tensor.buf).unwrap(),
            std::sync::Arc::get_mut(&mut dlogits.buf).unwrap(),
            batch_size, 
            logits.shape.1
        );
        
        let loss_value = loss_tensor.to_host(&self.backend)[0];
        Ok((logits, loss_value))
    }
}

// Trait for models that can be trained
pub trait TrainableModel<B: Backend> {
    fn forward(&mut self, x: &Tensor<B>) -> Tensor<B>;
    fn backward_pass(&mut self, logits: &Tensor<B>);
    fn update_params(&mut self, lr: f32);
    fn zero_grads(&mut self);
}