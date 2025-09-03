love it—here’s a clean, pattern-driven architecture for Option B (Rust + Metal) that’s easy to extend later to wgpu without a rewrite.

High-level layout (separation of concerns)

mnist-metal/
  crates/
    core/          # device-agnostic math + training abstractions
    gpu_metal/     # Metal-specific runtime, kernels, caches, pools
    bin/           # CLI app (train/eval/bench)

Core abstractions (where most patterns live)

1) Backend bridge (Bridge + Strategy)

Why: decouple what the model needs (matmul/relu/etc.) from how a device executes it.

// Bridge: Abstraction depends on Backend; Strategy: choose CPU or Metal at runtime.
pub trait Backend: Send + Sync + 'static {
  type Buf: Send + Sync;                 // storage handle (HostBuf, MetalBuf)
  fn alloc(&self, len: usize) -> Self::Buf;
  fn upload(&self, host: &[f32]) -> Self::Buf;
  fn download(&self, buf: &Self::Buf, out: &mut [f32]);

  // minimal op surface
  fn relu(&self, x: &Self::Buf, y: &mut Self::Buf, n: usize);
  fn gemm(&self, a:&Self::Buf,b:&Self::Buf,c:&mut Self::Buf,
          m:usize,n:usize,k:usize, alpha:f32,beta:f32, bias: Option<&Self::Buf>);
  fn softmax_xent(&self, logits:&Self::Buf, labels:&Self::Buf,
                  loss:&mut Self::Buf, dlogits:&mut Self::Buf,
                  batch:usize, classes:usize);
  // ...add what you actually need (im2col, maxpool, sgd_step, etc.)
}

	•	Strategy: pick CpuBackend or MetalBackend via CLI flag; both implement Backend.
	•	Bridge: the model only sees B: Backend; swapping backends doesn’t change model code.

2) Tensors & lazy transfers (Proxy)

Why: avoid accidental host↔device ping-pong.

pub struct Tensor<B: Backend> {
  pub buf: std::sync::Arc<B::Buf>,
  pub shape: (usize, usize), // expand as needed
}

	•	Proxy idea: provide Tensor::to_host() and Tensor::to_device() that only move when required; keep all intermediate tensors on device.

3) Layers & models (Composite + Builder)

Why: compose a network from reusable parts and construct it ergonomically.

pub trait Layer<B: Backend> {
  fn forward(&mut self, x: &Tensor<B>) -> Tensor<B>;
  fn backward(&mut self, dy: &Tensor<B>) -> Tensor<B>;
  fn params(&mut self) -> &[Tensor<B>];           // weights/biases
  fn grads(&mut self) -> &mut [Tensor<B>];        // grads (for optimizer)
}

pub struct Sequential<B: Backend> { pub layers: Vec<Box<dyn Layer<B>>>; }
impl<B: Backend> Layer<B> for Sequential<B> { /* Composite: iterate children */ }

pub struct ModelBuilder<B: Backend> { seq: Sequential<B> }      // Builder
impl<B: Backend> ModelBuilder<B> {
  pub fn conv2d(mut self, cfg: ConvCfg) -> Self { /* push layer */ self }
  pub fn relu(mut self) -> Self { /* ... */ self }
  pub fn maxpool2d(mut self, cfg: PoolCfg) -> Self { self }
  pub fn linear(mut self, in_:usize, out:usize) -> Self { self }
  pub fn build(self) -> Sequential<B> { self.seq }
}

	•	Composite: Sequential manages many Layers uniformly.
	•	Builder: fluent API to assemble LeNet/MLP variants for experiments.

4) Training orchestration (Template Method + Observer)

Why: keep one canonical loop with hook points and decouple metrics/logging.

pub struct Trainer<B: Backend> { /* ... */ }

pub trait Callbacks {
  fn on_epoch_begin(&mut self, _e: usize) {}
  fn on_batch_end(&mut self, _e: usize, _b: usize, _metrics: &Metrics) {}
  fn on_epoch_end(&mut self, _e: usize, _metrics: &Metrics) {}
}

impl<B: Backend> Trainer<B> {
  pub fn fit<L: Layer<B>>(&mut self, model: &mut L, data: &mut impl Iterator<Item=Batch<B>>,
                          cb: &mut impl Callbacks) {
    // Template Method: structure fixed, hooks fire via Observer
  }
}

	•	Template Method: fit defines the algorithm; subclasses aren’t needed—callbacks inject behavior.
	•	Observer: pluggable metrics/loggers (CSV writer, progress bar, profiler) implement Callbacks.

5) Optimizers (Strategy)

Why: interchangeable update policies.

pub trait Optimizer<B: Backend> {
  fn step(&mut self, params: &mut [Tensor<B>], grads: &mut [Tensor<B>]);
}
// SGD, SGD+Momentum, Adam as strategies

6) Data pipeline (Iterator)

Why: clean batch streaming with zero device awareness in core.

pub struct MnistLoader { /* mmap+normalize */ }
impl Iterator for MnistLoader { type Item = (Vec<f32>, Vec<u8>); /* ... */ }

Wrap with a small adaptor that uploads to device when backend is GPU.

7) Cross-cutting: timing/logging (Decorator)

Why: add profiling without touching kernels.
	•	Wrap a Backend in TimedBackend<B> that measures each call then forwards to the real backend.

pub struct TimedBackend<B: Backend> { inner: B, sink: MetricsSink }
impl<B: Backend> Backend for TimedBackend<B> { /* start/stop timers, delegate */ }

Metal runtime (gpu_metal crate)

8) Kernel registry & pipeline cache (Abstract Factory + Flyweight)

Why: compile once, reuse everywhere.
	•	Abstract Factory: KernelFactory produces typed “ops” (e.g., MatmulKernel, ReluKernel) from shader names + specialization constants (tile sizes).
	•	Flyweight: PipelineCache stores MTLComputePipelineState keyed by (shader, tile_m, tile_n, tile_k); shared across calls.

struct KernelFactory { device: metal::Device, cache: PipelineCache }
impl KernelFactory {
  fn matmul(&self, tiling: (u32,u32,u32)) -> MatmulKernel { /* lookup/create */ }
}

9) Buffer pool (Object Pool)

Why: avoid frequent MTLBuffer allocations.
	•	Keep a BTreeMap<usize, Vec<MTLBuffer>> keyed by capacity; “rent” on alloc, “return” on drop.

10) Launch helpers (Adapter)

Why: translate logical tensor shapes to Metal threadgroup geometry.
	•	Small adapters compute grid = ceil_div(M, TILE_M) × ceil_div(N, TILE_N) and bind resources.
	•	The Adapter layer isolates all Metal index math from core.

11) State (State)

Why: unify train/eval toggles (if you add dropout/batchnorm later).
	•	TrainMode vs EvalMode kept in Trainer and consulted by layers.

Pattern map (one-liner cheat sheet)
	•	Bridge: Backend trait decouples models from device.
	•	Strategy: pick CPU vs Metal backend; pick optimizer type.
	•	Composite: Sequential model of many Layers.
	•	Builder: ergonomic model construction.
	•	Template Method: canonical fit() loop.
	•	Observer: callbacks for metrics/progress.
	•	Iterator: data loader batches.
	•	Decorator: timing/logging around a backend.
	•	Adapter: shape→dispatch mapping for kernels.
	•	Abstract Factory: create kernels/pipelines.
	•	Flyweight: pipeline & shader cache; share compiled states.
	•	Object Pool: reuse MTLBuffers.
	•	Proxy: lazy H↔D moves through Tensor.
	•	State: train/eval behavior switches.

Anti-bloat guardrails (what not to over-engineer)
	•	Keep the Backend surface minimal (only ops you need).
	•	Don’t template everything—use a single generic B: Backend at the model level and trait objects for layers.
	•	Avoid deep inheritance—prefer composition (Composite/Decorator).
	•	Don’t allocate per batch: use BufferPool + in-place ops.

First implementation slice (order of work)
	1.	Backend CPU + CPU MLP (Template Method, Strategy, Composite, Builder).
	2.	MetalBackend skeleton (Bridge in place) + Relu kernel (Adapter + PipelineCache).
	3.	Tiled matmul + bias (Abstract Factory + Flyweight + Adapter).
	4.	Softmax+XEnt forward/dY, dReLU, dGEMM; add TimedBackend Decorator.
	5.	Optimizer Strategy (SGD+momentum).
	6.	im2col + matmul + MaxPool; add BufferPool.
	7.	Observer (CSV logger + progress) & State (train/eval switch).

Small, concrete type signatures (to keep you honest)

// core::tensor
impl<B: Backend> Tensor<B> {
  pub fn zeros(backend: &B, len: usize) -> Self { /* alloc + fill */ }
  pub fn from_host(backend: &B, data: &[f32], shape: (usize,usize)) -> Self { /* upload */ }
}

// core::layers
pub struct Linear<B: Backend> { w: Tensor<B>, b: Tensor<B>, /* caches for backward */ }
pub struct ReLU;
pub struct MaxPool2d { k: (usize,usize), s:(usize,usize) }

// core::optim
pub struct Sgd { lr: f32, momentum: Option<f32> }

// gpu_metal::backend
pub struct MetalBackend { dev: Device, queue: CommandQueue, factory: KernelFactory, pools: BufferPool }
impl Backend for MetalBackend { /* map ops to kernels */ }

How this future-proofs Option A (wgpu)
	•	The Bridge (Backend) and Adapter (shape→dispatch) keep the core device-agnostic.
	•	Implement WgpuBackend later that satisfies Backend with WGSL kernels; reuse all core code and most layers/Trainer.

If you want, I can turn this into a tiny repo skeleton (folders, trait stubs, and no-op CPU backend + Metal device initializer) so you can start filling in kernels immediately.