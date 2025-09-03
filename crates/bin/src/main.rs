use clap::{Parser, Subcommand};
use anyhow::Result;

use mnist_core::{
    backend::{Backend, CpuBackend},
    tensor::Tensor,
    nn::{MLP, LeNet, Sgd},
    data::{MnistDataset, MnistAugmentedDataset, LRScheduler},
    train::{Trainer, Metrics, Callbacks},
    profiler::PerformanceProfiler,
};

use mnist_gpu_metal::MetalBackend;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    CpuMlp {
        #[arg(long, default_value_t = 3)]
        epochs: usize,
        #[arg(long, default_value_t = 64)]
        batch: usize,
        #[arg(long, default_value_t = 0.01)]
        lr: f32,
    },
    GpuMlp {
        #[arg(long, default_value_t = 3)]
        epochs: usize,
        #[arg(long, default_value_t = 256)]
        batch: usize,
        #[arg(long, default_value_t = 0.01)]
        lr: f32,
    },
    GpuCnn {
        #[arg(long, default_value_t = 5)]
        epochs: usize,
        #[arg(long, default_value_t = 256)]
        batch: usize,
        #[arg(long, default_value_t = 0.01)]
        lr: f32,
    },
    /// M6 Enhanced CNN with all improvements (He init, profiling, etc.)
    M6Cnn {
        #[arg(long, default_value_t = 5)]
        epochs: usize,
        #[arg(long, default_value_t = 256)]
        batch: usize,
        #[arg(long, default_value_t = 0.005)]
        lr: f32,
        #[arg(long)]
        augmentation: bool,
        #[arg(long)]
        profiling: bool,
    },
    Eval {
        #[arg(long)]
        ckpt: String,
    },
    /// CPU↔GPU parity checks (softmax-xent, im2col)
    Parity {
        #[arg(long, default_value_t = 32)]
        batch: usize,
        #[arg(long, default_value_t = 10)]
        classes: usize,
    },
    /// Run tiny CPU baseline sanity checks
    Sanity {
        #[arg(long, default_value_t = 64)]
        batch: usize,
        #[arg(long, default_value_t = 200)]
        steps: usize,
        #[arg(long, default_value_t = 0.01)]
        lr: f32,
    },
}

struct CsvLogger {
    train_metrics: Vec<String>,
    timing_metrics: Vec<String>,
}

impl CsvLogger {
    fn new() -> Self {
        Self {
            train_metrics: vec!["epoch,batch,loss,accuracy".to_string()],
            timing_metrics: vec!["operation,time_ms,throughput_images_per_sec".to_string()],
        }
    }
    
    fn save_metrics(&self) -> Result<()> {
        std::fs::write("results/metrics_train.csv", self.train_metrics.join("\n"))?;
        std::fs::write("results/op_timings.csv", self.timing_metrics.join("\n"))?;
        println!("Saved metrics to results/");
        Ok(())
    }
}

impl Callbacks for CsvLogger {
    fn on_epoch_begin(&mut self, epoch: usize) {
        println!("Starting epoch {}", epoch + 1);
    }
    
    fn on_batch_end(&mut self, epoch: usize, batch: usize, metrics: &Metrics) {
        if batch % 100 == 0 {
            println!("Epoch {}, Batch {}: Loss={:.4}, Acc={:.2}%", 
                epoch + 1, batch, metrics.loss, metrics.accuracy * 100.0);
        }
        
        self.train_metrics.push(format!("{},{},{:.6},{:.4}", 
            epoch, batch, metrics.loss, metrics.accuracy));
    }
    
    fn on_epoch_end(&mut self, epoch: usize, metrics: &Metrics) {
        println!("Epoch {} complete: Loss={:.4}, Acc={:.2}%", 
            epoch + 1, metrics.loss, metrics.accuracy * 100.0);
    }
}

fn train_cpu_mlp(epochs: usize, batch_size: usize, learning_rate: f32) -> Result<()> {
    println!("Loading MNIST dataset...");
    let train_data = MnistDataset::load_train()?;
    let test_data = MnistDataset::load_test()?;
    
    println!("Initializing CPU backend and MLP model...");
    let backend = CpuBackend;
    let mut model = MLP::new(backend, &[784, 128, 10]);
    let _optimizer = Sgd::new(CpuBackend, learning_rate, Some(0.9));
    
    let trainer = Trainer::new(CpuBackend);
    let mut callbacks = CsvLogger::new();
    
    println!("Training MLP on CPU for {} epochs...", epochs);
    
    for epoch in 0..epochs {
        callbacks.on_epoch_begin(epoch);
        
        let mut epoch_loss = 0.0;
        let mut epoch_acc = 0.0;
        let mut num_batches = 0;
        
        for batch_start in (0..train_data.num_samples).step_by(batch_size) {
            let (batch_images, batch_labels) = train_data.get_batch(batch_start, batch_size);
            let actual_batch_size = batch_labels.len();
            
            // Forward pass
            let x = Tensor::from_host(&CpuBackend, &batch_images, (actual_batch_size, 784));
            let logits = model.forward(&x);
            
            // Compute loss and gradients
            let labels_f32: Vec<f32> = batch_labels.iter().map(|&l| l as f32).collect();
            let labels_tensor = Tensor::from_host(&CpuBackend, &labels_f32, (actual_batch_size, 1));
            
            let mut loss_tensor = Tensor::zeros(&CpuBackend, 1, 1);
            let mut grad_logits = Tensor::zeros(&CpuBackend, actual_batch_size, 10);
            
            CpuBackend.softmax_xent(&logits.buf, &labels_tensor.buf, 
                                   &mut std::sync::Arc::get_mut(&mut loss_tensor.buf).unwrap(),
                                   &mut std::sync::Arc::get_mut(&mut grad_logits.buf).unwrap(),
                                   actual_batch_size, 10);
            
            let loss = loss_tensor.to_host(&CpuBackend)[0];
            
            // Backward pass (ensure grads buffer starts at zero for accumulation)
            model.zero_grads();
            model.backward(&grad_logits);
            
            // Parameter update (SGD)
            model.update_params(learning_rate);
            
            let accuracy = trainer.compute_accuracy(&logits, &batch_labels);
            
            let metrics = Metrics {
                loss,
                accuracy,
                epoch,
                batch: num_batches,
            };
            
            callbacks.on_batch_end(epoch, num_batches, &metrics);
            
            epoch_loss += loss;
            epoch_acc += accuracy;
            num_batches += 1;
        }
        
        let avg_metrics = Metrics {
            loss: epoch_loss / num_batches as f32,
            accuracy: epoch_acc / num_batches as f32,
            epoch,
            batch: 0,
        };
        
        callbacks.on_epoch_end(epoch, &avg_metrics);
        
        // Test accuracy  
        if epoch % 1 == 0 {
            let (test_images, test_labels) = test_data.get_batch(0, 1000);
            let test_x = Tensor::from_host(&CpuBackend, &test_images, (test_labels.len(), 784));
            let test_logits = model.forward(&test_x);
            let test_acc = trainer.compute_accuracy(&test_logits, &test_labels);
            println!("Epoch {}: Test accuracy = {:.2}%", epoch + 1, test_acc * 100.0);
        }
    }
    
    callbacks.save_metrics()?;
    println!("Training complete! Model checkpoint would be saved to checkpoints/mlp_cpu.ckpt");
    
    Ok(())
}

fn train_gpu_mlp(epochs: usize, batch_size: usize, learning_rate: f32) -> Result<()> {
    println!("Loading MNIST dataset...");
    let train_data = MnistDataset::load_train()?;
    let _test_data = MnistDataset::load_test()?;
    
    println!("Initializing Metal backend...");
    let backend = MetalBackend::new()?;
    println!("Metal backend initialized successfully!");
    
    println!("Creating MLP model on GPU...");
    let mut model = MLP::new(backend.clone(), &[784, 128, 10]);
    
    let _trainer = Trainer::new(backend.clone());
    let mut callbacks = CsvLogger::new();
    
    println!("Training MLP on GPU for {} epochs...", epochs);
    
    for epoch in 0..epochs {
        callbacks.on_epoch_begin(epoch);
        
        let mut epoch_loss = 0.0;
        let mut epoch_acc = 0.0;
        let mut num_batches = 0;
        
        // Process a smaller number of batches for testing
        let max_batches = 100; // Limit for testing
        
        for batch_start in (0..train_data.num_samples).step_by(batch_size).take(max_batches) {
            let (batch_images, batch_labels) = train_data.get_batch(batch_start, batch_size);
            let actual_batch_size = batch_labels.len();
            
            // Forward pass on GPU
            let x = Tensor::from_host(&backend, &batch_images, (actual_batch_size, 784));
            let logits = model.forward(&x);
            
            // Compute loss and gradients on GPU
            let labels_f32: Vec<f32> = batch_labels.iter().map(|&l| l as f32).collect();
            let labels_tensor = Tensor::from_host(&backend, &labels_f32, (actual_batch_size, 1));
            
            let mut loss_tensor = Tensor::zeros(&backend, actual_batch_size, 1); // Per-sample losses
            let mut grad_logits = Tensor::zeros(&backend, actual_batch_size, 10);
            
            backend.softmax_xent(&logits.buf, &labels_tensor.buf,
                               std::sync::Arc::get_mut(&mut loss_tensor.buf).unwrap(),
                               std::sync::Arc::get_mut(&mut grad_logits.buf).unwrap(),
                               actual_batch_size, 10);
            
            let loss = loss_tensor.to_host(&backend)[0]; // Average loss computed by kernel
            
            // Compute accuracy on CPU for now
            let logits_cpu = logits.to_host(&backend);
            let mut correct = 0;
            for (i, &label) in batch_labels.iter().enumerate() {
                let start = i * 10;
                let end = start + 10;
                let predicted = logits_cpu[start..end].iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
                if predicted == label as usize {
                    correct += 1;
                }
            }
            let accuracy = correct as f32 / actual_batch_size as f32;
            
            // Backward pass on GPU
            model.zero_grads();
            model.backward(&grad_logits);
            
            // Simple parameter updates (could move to GPU later)
            model.update_params(learning_rate);
            
            let metrics = Metrics {
                loss,
                accuracy,
                epoch,
                batch: num_batches,
            };
            
            callbacks.on_batch_end(epoch, num_batches, &metrics);
            
            epoch_loss += loss;
            epoch_acc += accuracy;
            num_batches += 1;
        }
        
        let avg_metrics = Metrics {
            loss: epoch_loss / num_batches as f32,
            accuracy: epoch_acc / num_batches as f32,
            epoch,
            batch: 0,
        };
        
        callbacks.on_epoch_end(epoch, &avg_metrics);
        
        println!("Epoch {}: GPU forward pass successful! Loss={:.4}, Acc={:.2}%", 
            epoch + 1, avg_metrics.loss, avg_metrics.accuracy * 100.0);
    }
    
    callbacks.save_metrics()?;
    println!("GPU MLP testing complete!");
    
    Ok(())
}

fn train_gpu_cnn(epochs: usize, batch_size: usize, learning_rate: f32) -> Result<()> {
    println!("Loading MNIST dataset...");
    let train_data = MnistDataset::load_train()?;
    let _test_data = MnistDataset::load_test()?;
    
    println!("Initializing Metal backend...");
    let backend = MetalBackend::new()?;
    println!("Metal backend initialized successfully!");
    
    println!("Creating LeNet CNN model on GPU...");
    let mut model = LeNet::new(backend.clone());
    
    let _trainer = Trainer::new(backend.clone());
    let mut callbacks = CsvLogger::new();
    
    println!("Training LeNet CNN on GPU for {} epochs...", epochs);
    
    for epoch in 0..epochs {
        callbacks.on_epoch_begin(epoch);
        
        let mut epoch_loss = 0.0;
        let mut epoch_acc = 0.0;
        let mut num_batches = 0;
        
        // Process a limited number of batches for initial testing
        let max_batches = 50; // Limit for testing CNN implementation
        
        for batch_start in (0..train_data.num_samples).step_by(batch_size).take(max_batches) {
            let (batch_images, batch_labels) = train_data.get_batch(batch_start, batch_size);
            let actual_batch_size = batch_labels.len();
            
            // Forward pass on GPU
            let x = Tensor::from_host(&backend, &batch_images, (actual_batch_size, 784));
            let logits = model.forward(&x);
            
            // Compute loss and gradients on GPU
            let labels_f32: Vec<f32> = batch_labels.iter().map(|&l| l as f32).collect();
            let labels_tensor = Tensor::from_host(&backend, &labels_f32, (actual_batch_size, 1));
            
            let mut loss_tensor = Tensor::zeros(&backend, actual_batch_size, 1);
            let mut grad_logits = Tensor::zeros(&backend, actual_batch_size, 10);
            
            backend.softmax_xent(&logits.buf, &labels_tensor.buf,
                               std::sync::Arc::get_mut(&mut loss_tensor.buf).unwrap(),
                               std::sync::Arc::get_mut(&mut grad_logits.buf).unwrap(),
                               actual_batch_size, 10);
            
            let loss = loss_tensor.to_host(&backend)[0];
            
            // Compute accuracy on CPU for now
            let logits_cpu = logits.to_host(&backend);
            let mut correct = 0;
            for (i, &label) in batch_labels.iter().enumerate() {
                let start = i * 10;
                let end = start + 10;
                let predicted = logits_cpu[start..end].iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
                if predicted == label as usize {
                    correct += 1;
                }
            }
            let accuracy = correct as f32 / actual_batch_size as f32;
            
            // Backward pass on GPU
            model.zero_grads();
            model.backward(&grad_logits);
            
            // Parameter updates (simplified for now)
            model.update_params(learning_rate);
            
            let metrics = Metrics {
                loss,
                accuracy,
                epoch,
                batch: num_batches,
            };
            
            callbacks.on_batch_end(epoch, num_batches, &metrics);
            
            epoch_loss += loss;
            epoch_acc += accuracy;
            num_batches += 1;
        }
        
        let avg_metrics = Metrics {
            loss: epoch_loss / num_batches as f32,
            accuracy: epoch_acc / num_batches as f32,
            epoch,
            batch: 0,
        };
        
        callbacks.on_epoch_end(epoch, &avg_metrics);
        
        println!("Epoch {}: CNN forward pass successful! Loss={:.4}, Acc={:.2}%", 
            epoch + 1, avg_metrics.loss, avg_metrics.accuracy * 100.0);
    }
    
    callbacks.save_metrics()?;
    println!("GPU CNN training complete!");
    
    Ok(())
}

// M6 Enhanced CNN training with all improvements
fn train_m6_cnn(epochs: usize, batch_size: usize, learning_rate: f32, use_augmentation: bool, use_profiling: bool) -> Result<()> {
    println!("🚀 M6 Enhanced CNN Training with all improvements enabled!");
    
    // Load MNIST dataset
    println!("Loading MNIST dataset...");
    let train_data = MnistDataset::load_train()?;
    let test_data = MnistDataset::load_test()?;
    
    // Initialize Metal backend with error handling
    println!("Initializing Metal GPU backend...");
    let backend = MetalBackend::new()?;
    println!("✅ Metal backend initialized successfully!");
    
    // Create enhanced LeNet model with He initialization
    println!("Creating LeNet model with He initialization...");
    let mut model = LeNet::new_he_initialization(backend.clone());
    
    // Set up profiling if requested
    let profiler = if use_profiling {
        println!("✅ Performance profiling enabled");
        Some(PerformanceProfiler::new())
    } else {
        None
    };
    
    // Set up learning rate scheduler
    let scheduler = LRScheduler::new(learning_rate, 0.5, vec![3, 7]);
    
    // Create optimized trainer with profiling and scheduling
    let mut trainer = Trainer::new(backend.clone());
    if let Some(profiler) = profiler {
        trainer = trainer.with_profiler(profiler);
    }
    trainer = trainer.with_scheduler(scheduler);
    
    // Set up data augmentation if requested
    let augmenter = if use_augmentation {
        println!("✅ Data augmentation enabled (rotation, shift, noise)");
        Some(MnistAugmentedDataset::new(0.15, 0.1, 0.05)) // rotation, shift, noise
    } else {
        None
    };
    
    // Convert training data to the format expected by the optimized trainer
    let (train_images, train_labels) = train_data.get_batch(0, train_data.num_samples);
    let mut augmented_images = train_images.clone();
    
    if let Some(ref aug) = augmenter {
        println!("Applying data augmentation to training set...");
        aug.augment_batch(&mut augmented_images);
    }
    
    // Custom callbacks for enhanced logging
    struct EnhancedLogger {
        start_time: std::time::Instant,
        best_accuracy: f32,
    }
    
    impl EnhancedLogger {
        fn new() -> Self {
            Self {
                start_time: std::time::Instant::now(),
                best_accuracy: 0.0,
            }
        }
    }
    
    impl Callbacks for EnhancedLogger {
        fn on_epoch_begin(&mut self, epoch: usize) {
            println!("\n🔄 Starting epoch {} of enhanced training...", epoch + 1);
        }
        
        fn on_batch_end(&mut self, epoch: usize, batch: usize, metrics: &Metrics) {
            if batch % 50 == 0 {
                println!("  Epoch {}, Batch {}: Loss={:.4}, Acc={:.2}%", 
                    epoch + 1, batch, metrics.loss, metrics.accuracy * 100.0);
            }
        }
        
        fn on_epoch_end(&mut self, epoch: usize, metrics: &Metrics) {
            let elapsed = self.start_time.elapsed().as_secs_f32();
            if metrics.accuracy > self.best_accuracy {
                self.best_accuracy = metrics.accuracy;
                println!("🎯 New best accuracy: {:.2}%", self.best_accuracy * 100.0);
            }
            println!("✅ Epoch {} complete: Loss={:.4}, Acc={:.2}%, Time={:.1}s", 
                epoch + 1, metrics.loss, metrics.accuracy * 100.0, elapsed);
        }
    }
    
    let mut callbacks = EnhancedLogger::new();
    
    println!("\n🚀 Starting M6 enhanced training...");
    println!("Configuration:");
    println!("  • Epochs: {}", epochs);
    println!("  • Batch size: {}", batch_size);
    println!("  • Learning rate: {} (with scheduling)", learning_rate);
    println!("  • Data augmentation: {}", if use_augmentation { "enabled" } else { "disabled" });
    println!("  • Performance profiling: {}", if use_profiling { "enabled" } else { "disabled" });
    println!("  • He initialization: enabled");
    println!("  • Native GPU operations: enabled");
    println!("  • Im2Col integration: enabled");
    
    // Run the optimized training loop
    let training_start = std::time::Instant::now();
    let metrics_history = trainer.train_optimized(
        &mut model,
        &augmented_images,
        &train_labels,
        epochs,
        batch_size,
        learning_rate,
        &mut callbacks
    )?;
    let training_time = training_start.elapsed();
    
    println!("\n🎉 M6 Enhanced Training Complete!");
    println!("📊 Final Results:");
    
    if let Some(final_metrics) = metrics_history.last() {
        println!("  • Final training accuracy: {:.2}%", final_metrics.accuracy * 100.0);
        println!("  • Final training loss: {:.4}", final_metrics.loss);
    }
    
    // Test on held-out test set
    println!("\n🧪 Testing on MNIST test set...");
    let (test_images, test_labels) = test_data.get_batch(0, 1000);
    let test_x = Tensor::from_host(&backend, &test_images, (test_labels.len(), 784));
    let test_logits = model.forward(&test_x);
    let test_accuracy = trainer.compute_accuracy(&test_logits, &test_labels);
    
    println!("📈 Performance Summary:");
    println!("  • Test accuracy: {:.2}%", test_accuracy * 100.0);
    println!("  • Total training time: {:.2}s", training_time.as_secs_f32());
    println!("  • Average time per epoch: {:.2}s", training_time.as_secs_f32() / epochs as f32);
    println!("  • Training speed: {:.0} images/sec", 
        (train_labels.len() * epochs) as f32 / training_time.as_secs_f32());
    
    // Check if we achieved the M6 targets
    let target_accuracy = 98.5;
    let target_epoch_time = 30.0;
    let actual_epoch_time = training_time.as_secs_f32() / epochs as f32;
    
    println!("\n🎯 M6 Target Achievement:");
    if test_accuracy * 100.0 >= target_accuracy {
        println!("  ✅ Accuracy target: {:.2}% >= {:.1}% (ACHIEVED!)", test_accuracy * 100.0, target_accuracy);
    } else {
        println!("  ❌ Accuracy target: {:.2}% < {:.1}% (needs improvement)", test_accuracy * 100.0, target_accuracy);
    }
    
    if actual_epoch_time <= target_epoch_time {
        println!("  ✅ Speed target: {:.1}s <= {:.1}s per epoch (ACHIEVED!)", actual_epoch_time, target_epoch_time);
    } else {
        println!("  ⚠️  Speed target: {:.1}s > {:.1}s per epoch (could be improved)", actual_epoch_time, target_epoch_time);
    }
    
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    match args.command {
        Commands::CpuMlp { epochs, batch, lr } => {
            train_cpu_mlp(epochs, batch, lr)?;
        }
        Commands::GpuMlp { epochs, batch, lr } => {
            train_gpu_mlp(epochs, batch, lr)?;
        }
        Commands::GpuCnn { epochs, batch, lr } => {
            train_gpu_cnn(epochs, batch, lr)?;
        }
        Commands::M6Cnn { epochs, batch, lr, augmentation, profiling } => {
            train_m6_cnn(epochs, batch, lr, augmentation, profiling)?;
        }
        Commands::Eval { ckpt } => {
            println!("Model evaluation would load checkpoint: {}", ckpt);
        }
        Commands::Parity { batch, classes } => {
            run_parity(batch, classes)?;
        }
        Commands::Sanity { batch, steps, lr } => {
            run_sanity_cpu_mlp(batch, steps, lr)?;
        }
    }
    
    Ok(())
}

fn run_sanity_cpu_mlp(batch_size: usize, steps: usize, lr: f32) -> Result<()> {
    println!("Running sanity check: CPU MLP on one fixed batch");
    let train_data = mnist_core::data::MnistDataset::load_train()?;
    let (batch_images, batch_labels) = train_data.get_batch(0, batch_size);

    let backend = CpuBackend;
    let mut model = MLP::new(backend, &[784, 128, 10]);

    let x = Tensor::from_host(&CpuBackend, &batch_images, (batch_labels.len(), 784));
    let labels_f32: Vec<f32> = batch_labels.iter().map(|&l| l as f32).collect();
    let labels_tensor = Tensor::from_host(&CpuBackend, &labels_f32, (batch_labels.len(), 1));

    let mut last_loss: Option<f32> = None;
    for step in 0..steps {
        // Forward
        let logits = model.forward(&x);

        // Loss + grads
        let mut loss_tensor = Tensor::zeros(&CpuBackend, 1, 1);
        let mut grad_logits = Tensor::zeros(&CpuBackend, batch_labels.len(), 10);
        CpuBackend.softmax_xent(
            &logits.buf,
            &labels_tensor.buf,
            &mut std::sync::Arc::get_mut(&mut loss_tensor.buf).unwrap(),
            &mut std::sync::Arc::get_mut(&mut grad_logits.buf).unwrap(),
            batch_labels.len(),
            10,
        );
        let loss = loss_tensor.to_host(&CpuBackend)[0];

        // Backward + update
        model.zero_grads();
        model.backward(&grad_logits);
        model.update_params(lr);

        // Accuracy on this batch
        let probs = logits.to_host(&CpuBackend);
        let mut correct = 0;
        for (i, &label) in batch_labels.iter().enumerate() {
            let start = i * 10;
            let end = start + 10;
            let pred = probs[start..end]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            if pred == label as usize {
                correct += 1;
            }
        }
        let acc = correct as f32 / batch_labels.len() as f32;

        if step % 20 == 0 || step + 1 == steps {
            // Print gradient norms (dw/db) for visibility
            let mut norms = Vec::new();
            for g in model.grads() {
                let v = g.to_host(&CpuBackend);
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                norms.push(format!("{:.3}", n));
            }
            println!(
                "step {:>3} | loss {:.4} | acc {:>5.1}% | grad_norms [{}]",
                step,
                loss,
                acc * 100.0,
                norms.join(", ")
            );
        }

        if let Some(prev) = last_loss {
            // Quick monotonic-ish check
            if step > 0 && loss.is_finite() && prev.is_finite() && loss > prev + 1e-3 {
                // non-fatal: just signal if loss increases a lot
                // println!("note: loss increased: {:.4} -> {:.4}", prev, loss);
            }
        }
        last_loss = Some(loss);
    }

    println!("Sanity check complete.");
    Ok(())
}

fn run_parity(batch: usize, classes: usize) -> Result<()> {
    use mnist_gpu_metal::MetalBackend;

    println!("Running parity checks: batch={}, classes={}...", batch, classes);

    // Build deterministic logits and labels
    let mut logits = vec![0.0f32; batch * classes];
    for b in 0..batch {
        for c in 0..classes {
            let idx = b * classes + c;
            logits[idx] = (0.03 * (idx as f32).sin()) + (0.01 * (c as f32));
        }
    }
    let labels: Vec<u8> = (0..batch).map(|i| (i % classes) as u8).collect();

    // CPU compute
    let cpu_logits = Tensor::from_host(&CpuBackend, &logits, (batch, classes));
    let cpu_labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
    let cpu_labels = Tensor::from_host(&CpuBackend, &cpu_labels_f32, (batch, 1));
    let mut cpu_loss = Tensor::zeros(&CpuBackend, 1, 1);
    let mut cpu_dy = Tensor::zeros(&CpuBackend, batch, classes);
    CpuBackend.softmax_xent(
        &cpu_logits.buf,
        &cpu_labels.buf,
        &mut std::sync::Arc::get_mut(&mut cpu_loss.buf).unwrap(),
        &mut std::sync::Arc::get_mut(&mut cpu_dy.buf).unwrap(),
        batch,
        classes,
    );
    let cpu_loss_val = cpu_loss.to_host(&CpuBackend)[0];
    let cpu_dy_host = cpu_dy.to_host(&CpuBackend);

    // GPU compute
    let gpu = MetalBackend::new()?;
    let gpu_logits = Tensor::from_host(&gpu, &logits, (batch, classes));
    let gpu_labels = Tensor::from_host(&gpu, &cpu_labels_f32, (batch, 1));
    let mut gpu_loss = Tensor::zeros(&gpu, batch, 1); // per-sample losses, averaged in backend
    let mut gpu_dy = Tensor::zeros(&gpu, batch, classes);
    gpu.softmax_xent(
        &gpu_logits.buf,
        &gpu_labels.buf,
        &mut std::sync::Arc::get_mut(&mut gpu_loss.buf).unwrap(),
        &mut std::sync::Arc::get_mut(&mut gpu_dy.buf).unwrap(),
        batch,
        classes,
    );
    let gpu_loss_val = gpu_loss.to_host(&gpu)[0];
    let gpu_dy_host = gpu_dy.to_host(&gpu);

    // Compare
    let loss_diff = (cpu_loss_val - gpu_loss_val).abs();
    let grad_diff = cpu_dy_host
        .iter()
        .zip(gpu_dy_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |acc, x| acc.max(x));

    println!("softmax_xent: loss_cpu={:.6}, loss_gpu={:.6}, |∆|={:.6}", cpu_loss_val, gpu_loss_val, loss_diff);
    println!("softmax_xent: max|grad_cpu-gpu|={:.8}", grad_diff);

    let ok = loss_diff < 1e-5 && grad_diff < 1e-5;
    println!("Parity: {}", if ok { "PASS" } else { "FAIL" });

    // im2col tiny check: input 1x1x3x3, k=2, s=1, no pad
    let input_h = 3usize;
    let input_w = 3usize;
    let input: Vec<f32> = (0..(1*1*input_h*input_w)).map(|i| i as f32).collect();

    let cpu_in = Tensor::from_host(&CpuBackend, &input, (1, 1*input_h*input_w));
    let mut cpu_cols = Tensor::zeros(&CpuBackend, 1*2*2, (input_h-1)*(input_w-1)); // rows, cols
    CpuBackend.im2col(
        &cpu_in.buf,
        &mut std::sync::Arc::get_mut(&mut cpu_cols.buf).unwrap(),
        (1,1,input_h,input_w), (2,2), (0,0), (1,1), (input_h-1, input_w-1)
    );
    let cpu_cols_h = cpu_cols.to_host(&CpuBackend);

    let gpu_in = Tensor::from_host(&gpu, &input, (1, 1*input_h*input_w));
    let mut gpu_cols = Tensor::zeros(&gpu, 1*2*2, (input_h-1)*(input_w-1));
    gpu.im2col(
        &gpu_in.buf,
        &mut std::sync::Arc::get_mut(&mut gpu_cols.buf).unwrap(),
        (1,1,input_h,input_w), (2,2), (0,0), (1,1), (input_h-1, input_w-1)
    );
    let gpu_cols_h = gpu_cols.to_host(&gpu);

    let im2col_maxdiff = cpu_cols_h
        .iter()
        .zip(gpu_cols_h.iter())
        .map(|(a,b)| (a-b).abs())
        .fold(0.0f32, |acc, x| acc.max(x));
    println!("im2col tiny: max|cpu-gpu|={:.8} => {}", im2col_maxdiff, if im2col_maxdiff < 1e-5 { "PASS" } else { "FAIL" });

    Ok(())
}
