# Rusty Recognition - MNIST with Metal

A high-performance MNIST digit recognition project implemented in Rust with custom Metal GPU kernels for Apple Silicon.

## Project Structure

```
mnist-metal/
├── Cargo.toml                  # Workspace configuration
├── crates/
│   ├── core/                   # Device-agnostic ML abstractions
│   │   ├── src/
│   │   │   ├── backend.rs      # Backend trait and CPU implementation
│   │   │   ├── tensor.rs       # Tensor abstraction
│   │   │   ├── ops_cpu/        # CPU operations (matmul, relu, softmax)
│   │   │   ├── nn/             # Neural network layers and models
│   │   │   ├── data/           # MNIST data loading
│   │   │   └── train.rs        # Training orchestration
│   │   └── Cargo.toml
│   ├── gpu_metal/              # Metal GPU implementation
│   │   ├── src/
│   │   │   ├── device.rs       # Metal device and pipeline management
│   │   │   ├── buffer.rs       # GPU buffer management
│   │   │   ├── backend.rs      # Metal backend implementation
│   │   │   └── ops/            # GPU operation wrappers
│   │   ├── shaders/
│   │   │   └── kernels.metal   # Metal Shading Language kernels
│   │   └── Cargo.toml
│   └── bin/                    # CLI application
│       ├── src/main.rs         # Main CLI with training commands
│       └── Cargo.toml
├── data/                       # MNIST dataset files (download separately)
├── checkpoints/                # Saved model checkpoints
└── results/                    # Training metrics and timing data
```

## Architecture Patterns

The codebase implements several design patterns for clean, extensible ML infrastructure:

- **Backend Bridge**: Decouples models from compute devices (CPU/GPU)
- **Strategy**: Interchangeable backends and optimizers  
- **Composite**: Sequential models built from reusable layers
- **Builder**: Fluent API for model construction
- **Template Method**: Standardized training loop with customizable callbacks
- **Observer**: Pluggable metrics collection and logging

## Features

- ✅ CPU baseline implementation with naive operations
- ✅ Metal GPU backend with custom MSL kernels
- ✅ Modular tensor abstraction supporting multiple backends
- ✅ MLP (Multi-Layer Perceptron) implementation
- 🚧 CNN (Convolutional Neural Network) with im2col + tiled matmul
- 🚧 Optimized tiled matrix multiplication kernels
- 🚧 Fused softmax + cross-entropy kernels
- ✅ SGD optimizer with momentum support
- ✅ CLI with training and evaluation modes

## Quick Start

### Prerequisites

- Rust 1.75+ with Cargo
- macOS with Apple Silicon (for Metal support)
- MNIST dataset files (see Data Setup below)

### Build

```bash
cargo build --release
```

### Data Setup

Download MNIST dataset files and place in `data/` directory:
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`  
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

### Training Commands

```bash
# CPU MLP training
cargo run --release -- cpu-mlp --epochs 3 --batch 64 --lr 0.01

# GPU MLP training  
cargo run --release -- gpu-mlp --epochs 3 --batch 256 --lr 0.01

# GPU CNN training
cargo run --release -- gpu-cnn --epochs 5 --batch 256 --lr 0.01

# Model evaluation
cargo run --release -- eval --ckpt checkpoints/cnn_best.ckpt
```

## Performance Goals

- **Accuracy**: ≥97% test accuracy (CPU MLP), ≥98.5% (GPU CNN)
- **Speedup**: ≥5× GPU inference vs CPU for batch≥256
- **Training**: ≥2× GPU vs CPU epoch time

## Development Status

This project is currently in **Milestone M0** (Repo Scaffold). The basic structure is complete and ready for implementation of CPU operations and Metal kernels.

### Next Steps (M1 - CPU Baseline)
1. Implement CPU matrix multiplication and neural network training
2. Add MNIST data loading and preprocessing  
3. Create working MLP that achieves ≥97% accuracy
4. Set up training metrics collection

### Planned Milestones
- **M1**: CPU baseline MLP training
- **M2**: Metal GPU runtime and simple kernels (ReLU)
- **M3**: Tiled matrix multiplication on GPU
- **M4**: Full GPU backpropagation pipeline
- **M5**: CNN implementation with im2col convolution
- **M6**: Performance optimization and demo polish

## Metal Kernels

Custom MSL kernels implemented:
- `relu_forward` / `drelu`: Element-wise ReLU activation
- `naive_gemm`: Basic matrix multiplication
- `matmul_tiled`: Optimized tiled GEMM with threadgroup memory
- `softmax_xent`: Fused softmax + cross-entropy loss
- `sgd_step`: SGD parameter updates with momentum

## License

MIT OR Apache-2.0