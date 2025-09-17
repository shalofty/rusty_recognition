# Rust + Metal (MSL) Project Vision

A pragmatic, showable plan you can execute on your M4 Max, with clear milestones, acceptance criteria, and demo artifacts.

## Project Goal

Train and run inference for MNIST with your own Metal compute kernels (MSL), launched from Rust via metal-rs. Start with an MLP, then a LeNet-ish CNN using either direct conv or im2col+matmul. Keep a CPU path for correctness + benchmarks.

## Success Criteria (What "Done" Looks Like)

### Accuracy
- **CPU MLP (784→128→10)**: ≥97% test accuracy
- **Metal GPU CNN (LeNet-ish)**: ≥98.5% test accuracy

### Speedup
- **GPU inference vs single-thread CPU**: ≥5× for batch≥256
- **GPU training epoch**: ≥2× vs CPU baseline

### Evidence
- **Reproducible runs** (`cargo run …`)
- **Per-op timing table** (CSV) + two small charts (loss/epoch; images/sec)

## Tech Constraints

### Do Use
- Rust stable, metal/metal-rs, objc2 (transitively), bytemuck, thiserror/anyhow, clap, rand, tiny CSV/chart helper (or dump CSV; plot later).

### Don't Use
- cuDNN/cuBLAS, PyTorch, burn, candle, etc. (OK to use only for sanity checks, not in the main path.)

---

## Milestones & Timeline (10–12 Focused Days)

### M0 — Repo Scaffold (½ day)

- Cargo workspace with core (tensors/ops), gpu_metal (kernels + runtime), bin (CLI).
- Add CI to build on macOS (no GPU tests, just compile shaders & CPU tests).

**Deliverable**: builds on your Mac, `cargo test` (CPU stubs pass).

---

### M1 — CPU Baseline (1 day)

- Parse raw MNIST IDX files, normalize to f32 in [0,1].
- CPU ops: matmul (naive), relu, softmax, xent, SGD (+momentum).
- MLP training to ≥97% on test set. Save `mlp_cpu.ckpt`.

**Acceptance**: accuracy hit + loss curve PNG/CSV.

---

### M2 — Metal Plumbing (1 day)

- Create Device, CommandQueue, Library from in-repo .metal files.
- Buffer helpers: device allocation, upload/download, constant/uniform buffers.
- First kernel: relu_forward (1D).
- Add a tiny numerical parity test (CPU vs GPU relu on random data).

**Acceptance**: relu matches within 1e-6; timing printed for one dispatch.

---

### M3 — Tiled Matmul (Forward) (2 days)

- MSL kernel `matmul_tiled.metal`
- Threadgroup memory tiles Asub, Bsub (e.g., 16×16 or 32×8 depending on occupancy).
- Loop over K; `threadgroup_barrier(mem_flags::mem_threadgroup)` between phases.
- Bias add in a separate elemental kernel (or fuse later).
- Rust wrappers compute grid sizes from (M, N, K), bind buffers, dispatch.
- MLP forward inference on GPU (matmul → bias → relu → softmax).

**Acceptance**: GPU forward logits match CPU within tolerance; forward pass ≥5× faster for batch≥256.

---

### M4 — Backprop on GPU (2 days)

#### Kernels
- `softmax_xent_forward + dsoftmax_xent` (fused: write dY = Y - onehot(labels))
- `drelu` (mask or branchless)
- `matmul_grad` (reuse forward kernel with transposed args or implement C = A^T * B / C = A * B^T)
- `sgd_step` (elementwise update)

#### Full MLP Training on GPU
- Forward + backward.
- Log images/sec, per-op timings via `MTLCommandBuffer.GPUEndTime - GPUStartTime` (or wall timers around commit + waitUntilCompleted).

**Acceptance**: GPU MLP reaches same accuracy as CPU; ≥2× epoch speedup; timing table printed.

---

### M5 — CNN via Im2col (2 days)

- Implement im2col kernel (NCHW → columns) and reuse `matmul_tiled` for convolution.
- Add `maxpool2d_forward` (+ optional indices for exact backward; or recompute argmax per-window).
- **LeNet-ish**: Conv(1→8, 5×5) -> ReLU -> Pool(2×2) -> Conv(8→16, 5×5) -> ReLU -> Pool(2×2) -> FC -> Softmax.
- Train on GPU to ≥98.5% test accuracy.

**Acceptance**: accuracy met; benchmark sheet updated (CPU vs GPU inference/training).

---

### M6 — Polish & Demo Pack (1 day)

#### CLI
```bash
cargo run --release -- cpu-mlp --epochs 3
cargo run --release -- gpu-mlp --epochs 3 --batch 256
cargo run --release -- gpu-cnn --epochs 5 --batch 256
cargo run --release -- eval --ckpt checkpoints/cnn_best.ckpt
```

#### Outputs
- Save CSVs: `metrics_train.csv`, `op_timings.csv`.
- README: architecture notes, kernel tiling diagrams, how to run, results.
- Optional: checkpoint loader to print a confusion matrix.

**Acceptance**: one-command reproducibility + clean README.

---

## Code Structure

```
mnist-metal/
  Cargo.toml
  crates/
    core/
      src/
        tensor.rs          # host tensors (contiguous), shapes/strides
        ops_cpu/*.rs       # baseline CPU ops
        nn/{layers.rs, model_mlp.rs, model_cnn.rs, optim.rs}
        data/mnist.rs
        train.rs
    gpu_metal/
      src/
        device.rs          # Device, Queue, Library, Pipeline cache
        buffer.rs          # upload/download wrappers (bytemuck)
        launch.rs          # grid/block helpers, encoder builders
        ops/
          relu.rs
          matmul.rs
          im2col.rs
          reduce.rs
          softmax.rs
          pool.rs
      shaders/
        relu.metal
        matmul_tiled.metal
        im2col.metal
        reduce_row_max_sum.metal
        softmax_xent.metal
        drelu.metal
        matmul_grad.metal
        sgd_step.metal
    bin/
      src/main.rs          # CLI (clap), runs train/eval/bench commands
  checkpoints/
  results/                 # CSV + plots
```

---

## Kernel Sketches (MSL)

### relu_forward.metal

```metal
#include <metal_stdlib>
using namespace metal;

kernel void relu_forward(
  device const float* x   [[buffer(0)]],
  device float*       y   [[buffer(1)]],
  constant uint&      n   [[buffer(2)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid < n) {
    float v = x[gid];
    y[gid] = v > 0.f ? v : 0.f;
  }
}
```

### matmul_tiled.metal (Shape Idea)

- **Threadgroup size**: `threadgroup_size(16,16,1)`
- Each thread computes one C element in a tile
- Use `threadgroup float Asub[T*T];` and `Bsub[T*T];`
- Loop over K in chunks of T; load A/B tiles; barrier; accumulate; write C.

*(You'll implement this during M3; keep indices row-major and avoid bank conflicts by padding if needed.)*

---

## Testing & Validation

### Unit Tests (CPU)
- matmul correctness, softmax+loss stability, gradient checks via finite differences on tiny tensors.

### Parity Tests (GPU vs CPU)
- relu, matmul (small M,N,K), softmax_xent forward, drelu, matmul_grad on random tensors; assert max abs diff < 1e-4.

### Determinism
- Fix RNG seed; consistent shuffling.

### Numerical Safety
- Clip logits before exp; add 1e-7 eps in log; watch for overflow in reductions.

---

## Performance Notes (Apple GPU)

- Prefer larger batches (≥256).
- Measure and tune tile size for matmul; try (16,16), (32,8), (8,32).
- Minimize buffer rebindings; reuse pipelines.
- Consider fusing bias+relu after you're correct.
- Use CommandBuffer timestamps for GPU time; also report wall-clock end-to-end.

---

## Risks & Mitigations

### Shader Indexing Bugs
- Start with tiny shapes; add asserts on bounds; dump debug slices.

### Numerical Mismatch
- Test on small inputs; compare CPU/GPU layer-by-layer.

### Throughput Disappointment
- Check batch size, tile size, and avoid extra device↔host copies; keep data resident on GPU across layers.

---

## Stretch Goals (Nice-to-have)

- Direct conv kernel (no im2col), shared-memory tiling for 5×5.
- Mixed precision (FP16 storage, FP32 accum) using half.
- Kernel fusion: softmax+loss+dY, bias+relu.
- Simple profiler view that prints % time per op over an epoch.

---

## Demo Script (What You'll Show)

1. **Quick slide**: goal, design, kernels written, why Metal.
2. **Terminal**:
   ```bash
   cargo run --release -- cpu-mlp --epochs 1  # (show loss/acc)
   cargo run --release -- gpu-mlp --epochs 1 --batch 512  # (show speedup)
   cargo run --release -- gpu-cnn --epochs 3 --batch 512  # (hit ≥98.5%)
   ```
3. **Show** `op_timings.csv` and a tiny plot: per-op times and overall images/sec.
4. **Short "what I learned"** on tiling, threadgroup memory, and Apple GPU quirks.

---

*If you want, I can generate the repo skeleton (Cargo layout + metal-rs device init + working relu.metal + dispatch wrapper + CPU MLP baseline) so you can start at M2 immediately and fill in matmul next.*