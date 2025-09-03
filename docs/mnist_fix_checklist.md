MNIST Rust + Metal: Accuracy Fix Checklist

- Stage 0 CPU Baseline:
  - Model: MLP 784→128→10 with ReLU.
  - Data: One fixed MNIST batch (e.g., 64), normalized [0,1].
  - Loop: 200 steps SGD, fixed seed, no shuffling.
  - Assert: loss drops; accuracy > 50% on that batch.

- Instrumentation:
  - Print L2 norms for dw/db after backward.
  - Print Δparam norms after update (non‑zero).

- Loss/Label Parity:
  - Compare CPU softmax‑xent vs GPU kernel on same logits/labels (|diff| < 1e-5).
  - Ensure labels are 1D class indices length=batch; avoid double averaging.

- Shapes/Layouts:
  - Enforce NCHW for conv inputs; keep explicit reshape helpers.
  - Unit‑test im2col on 1×1×3×3 with 2×2 kernel, stride 1.

- Pooling Backward:
  - Keep argmax indices on device; verify gradient scatter to max positions.

- Learning‑Rate Sweep:
  - Try 1e‑3, 3e‑3, 1e‑2, 3e‑2 on the tiny CPU run.

- Guardrails:
  - After backward: finite, non‑zero grad norms.
  - After update: non‑zero Δparam norms.

Once accuracy is fixed:
- Pre‑allocate activation/grad pools.
- Fuse hot ops (e.g., Linear+ReLU).
- Batch size scan for Metal occupancy.
- Consider FP16 forward with FP32 master weights.

