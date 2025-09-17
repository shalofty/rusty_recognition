# Current Status Summary

Here's a concise wrap-up of what we did and where you're headed next.

## Core Fixes

### Backend Loss/Grad Math: Aligned CPU Softmax-Xent with GPU

- **Batch scaling**: scale dlogits by 1/batch (fixes exploding updates).
- **Numerics**: add epsilon inside log to avoid inf.
- **File**: `crates/core/src/backend.rs`

### Linear Backward Correctness: Eliminated Implicit Transpose Pitfalls

- Implemented explicit host-side loops for dW, db, dX to guarantee correctness across backends.
- **File**: `crates/core/src/nn/layers.rs`

### Real Parameter Updates: Made Updates Actually Mutate Params

- Added `update_params(lr)` to Layer and implemented for Linear/Conv2d; ReLU/MaxPool are no-ops.
- **Files**: 
  - `crates/core/src/nn/layers.rs`
  - `crates/core/src/nn/layers.rs`
  - `crates/core/src/nn/layers.rs`
  - `crates/core/src/nn/layers.rs`
- MLP now calls per-layer updates.
- **File**: `crates/core/src/nn/model_mlp.rs`

### Training Loop Hygiene: Correct Zeroing Order for Grads

- Zero grads before backward; then update.
- **File**: `crates/bin/src/main.rs`

## Sanity + Parity Harness

### Sanity Subcommand (CPU MLP, Fixed Batch)

- **Command**: `cargo run --package mnist-runner -- sanity --batch 64 --steps 200 --lr 0.01`
- **Result**: Loss drops smoothly; accuracy surpasses 50%; with 1000 steps it reached 100% on the fixed batch.
- **File**: `crates/bin/src/main.rs`

### Parity Subcommand (CPU↔GPU)

- Validates softmax-xent loss/grad and a tiny im2col case.
- **Command**: `cargo run --package mnist-runner -- parity --batch 16 --classes 10`
- **Result**: PASS with loss |∆|=0 and max grad diff = 0; im2col max diff = 0.
- **File**: `crates/bin/src/main.rs`

## GPU Pooling Index Consistency

- MaxPool kernel now stores argmax indices as float to match host f32 buffers.
- **File**: `crates/gpu_metal/shaders/kernels.metal`

## Docs + Cleanup

### Checklist Added for Accuracy Fixes and Validation Steps

- **File**: `docs/mnist_fix_checklist.md`

### CLI Warnings Tidied

- Unused imports/vars underscored.
- **Files**:
  - `crates/bin/src/main.rs`
  - `crates/bin/src/main.rs`
  - `crates/bin/src/main.rs`
  - `crates/bin/src/main.rs`

## What We Proved

- Loss/label/grad/update wiring is now correct.
- LR=0.01 is stable for the CPU MLP sanity; gradients are finite and decay.
- CPU↔GPU parity for softmax-xent and im2col is exact.

## Next Actions

### CPU MLP Train/Val

```bash
cargo run --package mnist-runner -- cpu-mlp --epochs 5 --batch 128 --lr 0.02
```

Expect ≥97% test accuracy; sweep lr ∈ {0.01, 0.02, 0.03} if needed.

### GPU MLP Parity-by-Slices

Already validated softmax + im2col. Run gpu-mlp for 1 epoch and compare batch loss curve to CPU.

### CNN on CPU First

Train LeNet; expect 98–99% test accuracy. Watch per-layer grad norms and NCHW→im2col shapes.

### Move CNN to GPU in Slices

im2col+GEMM → pool → full model; compare against CPU baseline at each step.