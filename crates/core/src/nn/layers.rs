use crate::{backend::Backend, tensor::Tensor};
use rand::{thread_rng, Rng};

pub trait Layer<B: Backend> {
    fn forward(&mut self, x: &Tensor<B>) -> Tensor<B>;
    fn backward(&mut self, dy: &Tensor<B>) -> Tensor<B>;
    fn params(&self) -> Vec<&Tensor<B>>;
    fn grads(&mut self) -> Vec<&mut Tensor<B>>;
    fn zero_grads(&mut self);
    fn update_params(&mut self, _lr: f32) {}
}

pub struct Linear<B: Backend> {
    pub w: Tensor<B>,
    pub b: Tensor<B>,
    pub dw: Tensor<B>,
    pub db: Tensor<B>,
    pub last_input: Option<Tensor<B>>,
    backend: B,
}

impl<B: Backend + Clone> Linear<B> {
    pub fn new(backend: B, in_features: usize, out_features: usize) -> Self {
        let w_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| thread_rng().gen_range(-0.1..0.1))
            .collect();
        let b_data = vec![0.0; out_features];
        
        Self {
            w: Tensor::from_host(&backend, &w_data, (in_features, out_features)),
            b: Tensor::from_host(&backend, &b_data, (1, out_features)),
            dw: Tensor::zeros(&backend, in_features, out_features),
            db: Tensor::zeros(&backend, 1, out_features),
            last_input: None,
            backend,
        }
    }
}

impl<B: Backend + Clone> Layer<B> for Linear<B> {
    fn forward(&mut self, x: &Tensor<B>) -> Tensor<B> {
        self.last_input = Some(x.clone());
        let (batch_size, in_features) = x.shape;
        let (_, out_features) = self.w.shape;
        
        let mut output = Tensor::zeros(&self.backend, batch_size, out_features);
        
        self.backend.gemm(
            &x.buf, &self.w.buf, Arc::get_mut(&mut output.buf).unwrap(),
            batch_size, out_features, in_features,
            1.0, 0.0, Some(&self.b.buf)
        );
        
        output
    }
    
    fn backward(&mut self, dy: &Tensor<B>) -> Tensor<B> {
        let x = self.last_input.as_ref().expect("Must call forward before backward");
        let (batch_size, in_features) = x.shape;
        let (_, out_features) = dy.shape;

        // Compute on host to ensure correct transposes without backend flags.
        let x_h = x.to_host(&self.backend);            // (batch, in)
        let dy_h = dy.to_host(&self.backend);          // (batch, out)
        let w_h = self.w.to_host(&self.backend);       // (in, out)

        // dW = X^T * dY  => (in, out)
        let mut dw_h = vec![0.0f32; in_features * out_features];
        for i in 0..in_features {
            for j in 0..out_features {
                let mut sum = 0.0f32;
                for b in 0..batch_size {
                    sum += x_h[b * in_features + i] * dy_h[b * out_features + j];
                }
                dw_h[i * out_features + j] = sum;
            }
        }
        self.dw = Tensor::from_host(&self.backend, &dw_h, (in_features, out_features));

        // db = sum_b(dY)  => (1, out)
        let mut db_h = vec![0.0f32; out_features];
        for b in 0..batch_size {
            for j in 0..out_features {
                db_h[j] += dy_h[b * out_features + j];
            }
        }
        self.db = Tensor::from_host(&self.backend, &db_h, (1, out_features));

        // dX = dY * W^T  => (batch, in)
        let mut dx_h = vec![0.0f32; batch_size * in_features];
        for b in 0..batch_size {
            for i in 0..in_features {
                let mut sum = 0.0f32;
                for j in 0..out_features {
                    sum += dy_h[b * out_features + j] * w_h[i * out_features + j];
                }
                dx_h[b * in_features + i] = sum;
            }
        }
        Tensor::from_host(&self.backend, &dx_h, (batch_size, in_features))
    }
    
    fn params(&self) -> Vec<&Tensor<B>> {
        vec![&self.w, &self.b]
    }
    
    fn grads(&mut self) -> Vec<&mut Tensor<B>> {
        vec![&mut self.dw, &mut self.db]
    }
    
    fn zero_grads(&mut self) {
        // Zero out gradients - we'll need to handle this per backend type
        // For now, create new zero tensors
        self.dw = Tensor::zeros(&self.backend, self.dw.shape.0, self.dw.shape.1);
        self.db = Tensor::zeros(&self.backend, self.db.shape.0, self.db.shape.1);
    }

    fn update_params(&mut self, lr: f32) {
        // Update parameters using simple SGD on the active backend
        self.w.sgd_update(&self.dw, &self.backend, lr);
        self.b.sgd_update(&self.db, &self.backend, lr);
    }
}

impl<B: Backend + Clone> Linear<B> { }

pub struct ReLU<B: Backend> {
    last_input: Option<Tensor<B>>,
    backend: B,
}

impl<B: Backend + Clone> ReLU<B> {
    pub fn new(backend: B) -> Self {
        Self {
            last_input: None,
            backend,
        }
    }
}

impl<B: Backend + Clone> Layer<B> for ReLU<B> {
    fn forward(&mut self, x: &Tensor<B>) -> Tensor<B> {
        self.last_input = Some(x.clone());
        let mut output = Tensor::zeros(&self.backend, x.shape.0, x.shape.1);
        
        self.backend.relu(&x.buf, Arc::get_mut(&mut output.buf).unwrap(), x.len());
        output
    }
    
    fn backward(&mut self, dy: &Tensor<B>) -> Tensor<B> {
        let x = self.last_input.as_ref().expect("Must call forward before backward");
        let mut dx = Tensor::zeros(&self.backend, x.shape.0, x.shape.1);
        
        self.backend.drelu(&dy.buf, &x.buf, Arc::get_mut(&mut dx.buf).unwrap(), x.len());
        dx
    }
    
    fn params(&self) -> Vec<&Tensor<B>> {
        vec![]
    }
    
    fn grads(&mut self) -> Vec<&mut Tensor<B>> {
        vec![]
    }
    
    fn zero_grads(&mut self) {
        // ReLU has no parameters, so no gradients to zero
    }

    fn update_params(&mut self, _lr: f32) {
        // ReLU has no parameters
    }
}

use std::sync::Arc;

pub struct Conv2d<B: Backend> {
    pub w: Tensor<B>, // (out_channels, in_channels, kernel_h, kernel_w)
    pub b: Tensor<B>, // (out_channels,)
    pub dw: Tensor<B>,
    pub db: Tensor<B>,
    pub last_input: Option<Tensor<B>>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub padding: (usize, usize),
    pub stride: (usize, usize),
    backend: B,
}

impl<B: Backend + Clone> Conv2d<B> {
    pub fn new(backend: B, in_channels: usize, out_channels: usize, 
               kernel_size: (usize, usize), padding: (usize, usize), stride: (usize, usize)) -> Self {
        let (kernel_h, kernel_w) = kernel_size;
        let total_kernel_size = out_channels * in_channels * kernel_h * kernel_w;
        
        // Xavier initialization
        let fan_in = in_channels * kernel_h * kernel_w;
        let fan_out = out_channels * kernel_h * kernel_w;
        let std_dev = (2.0 / (fan_in + fan_out) as f32).sqrt();
        
        let w_data: Vec<f32> = (0..total_kernel_size)
            .map(|_| thread_rng().gen_range(-std_dev..std_dev))
            .collect();
        let b_data = vec![0.0; out_channels];
        
        Self {
            w: Tensor::from_host(&backend, &w_data, (out_channels, in_channels * kernel_h * kernel_w)),
            b: Tensor::from_host(&backend, &b_data, (1, out_channels)),
            dw: Tensor::zeros(&backend, out_channels, in_channels * kernel_h * kernel_w),
            db: Tensor::zeros(&backend, 1, out_channels),
            last_input: None,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            backend,
        }
    }
    
    // M6 Improvement: He initialization for ReLU networks
    pub fn new_he_initialization(backend: B, in_channels: usize, out_channels: usize, 
                               kernel_size: (usize, usize), padding: (usize, usize), stride: (usize, usize)) -> Self {
        let (kernel_h, kernel_w) = kernel_size;
        let total_kernel_size = out_channels * in_channels * kernel_h * kernel_w;
        
        // He initialization - optimized for ReLU activations
        let fan_in = in_channels * kernel_h * kernel_w;
        let std_dev = (2.0 / fan_in as f32).sqrt();
        
        let w_data: Vec<f32> = (0..total_kernel_size)
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1: f32 = thread_rng().gen();
                let u2: f32 = thread_rng().gen();
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                std_dev * z0
            })
            .collect();
        let b_data = vec![0.0; out_channels];
        
        Self {
            w: Tensor::from_host(&backend, &w_data, (out_channels, in_channels * kernel_h * kernel_w)),
            b: Tensor::from_host(&backend, &b_data, (1, out_channels)),
            dw: Tensor::zeros(&backend, out_channels, in_channels * kernel_h * kernel_w),
            db: Tensor::zeros(&backend, 1, out_channels),
            last_input: None,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            backend,
        }
    }
    
    fn compute_output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let (kernel_h, kernel_w) = self.kernel_size;
        let (pad_h, pad_w) = self.padding;
        let (stride_h, stride_w) = self.stride;
        
        let output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
        let output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
        
        (output_h, output_w)
    }
}

impl<B: Backend + Clone> Layer<B> for Conv2d<B> {
    fn forward(&mut self, x: &Tensor<B>) -> Tensor<B> {
        self.last_input = Some(x.clone());
        
        // Assume input shape is (batch_size, in_channels * input_h * input_w)
        // We need to reshape this to (batch_size, in_channels, input_h, input_w)
        let total_elements = x.len();
        let batch_size = x.shape.0;
        let spatial_elements = total_elements / batch_size;
        let input_h = ((spatial_elements / self.in_channels) as f32).sqrt() as usize;
        let input_w = input_h; // Assume square images for MNIST
        
        let (output_h, output_w) = self.compute_output_size(input_h, input_w);
        let (kernel_h, kernel_w) = self.kernel_size;
        
        // M6 Improvement: Proper Im2Col Integration
        // Create im2col matrix: (kernel_h * kernel_w * in_channels, output_h * output_w * batch_size)
        let col_rows = kernel_h * kernel_w * self.in_channels;
        let col_cols = output_h * output_w * batch_size;
        let mut col_matrix = Tensor::zeros(&self.backend, col_rows, col_cols);
        
        // Apply im2col transformation
        self.backend.im2col(
            &x.buf,
            Arc::get_mut(&mut col_matrix.buf).unwrap(),
            (batch_size, self.in_channels, input_h, input_w),
            self.kernel_size,
            self.padding,
            self.stride,
            (output_h, output_w)
        );
        
        // Perform GEMM: W × im2col_matrix = output
        // W: (out_channels, in_channels * kernel_h * kernel_w)
        // col_matrix: (in_channels * kernel_h * kernel_w, output_h * output_w * batch_size)
        // output: (out_channels, output_h * output_w * batch_size)
        let mut temp_output = Tensor::zeros(&self.backend, self.out_channels, col_cols);
        
        self.backend.gemm(
            &self.w.buf, &col_matrix.buf, Arc::get_mut(&mut temp_output.buf).unwrap(),
            self.out_channels, col_cols, col_rows,
            1.0, 0.0, None
        );
        
        // Add bias and reshape output
        let mut output = Tensor::zeros(&self.backend, batch_size, self.out_channels * output_h * output_w);
        
        // Reshape and add bias: (out_channels, output_h * output_w * batch_size) -> (batch_size, out_channels * output_h * output_w)
        let temp_data = temp_output.to_host(&self.backend);
        let mut output_data = vec![0.0f32; batch_size * self.out_channels * output_h * output_w];
        let bias_data = self.b.to_host(&self.backend);
        
        for b in 0..batch_size {
            for c in 0..self.out_channels {
                for spatial in 0..(output_h * output_w) {
                    let temp_idx = c * (output_h * output_w * batch_size) + b * (output_h * output_w) + spatial;
                    let out_idx = b * (self.out_channels * output_h * output_w) + c * (output_h * output_w) + spatial;
                    output_data[out_idx] = temp_data[temp_idx] + bias_data[c];
                }
            }
        }
        
        output.buf = Arc::new(self.backend.upload(&output_data));
        output
    }
    
    fn backward(&mut self, dy: &Tensor<B>) -> Tensor<B> {
        let x = self.last_input.as_ref().expect("Must call forward before backward");
        
        // Simplified backward pass - proper implementation would use gradient kernels
        let mut dx = Tensor::zeros(&self.backend, x.shape.0, x.shape.1);
        
        // Weight gradients: dW = im2col(X)^T * dY (reshaped)
        self.backend.gemm(
            &x.buf, &dy.buf, Arc::get_mut(&mut self.dw.buf).unwrap(),
            self.dw.shape.0, self.dw.shape.1, x.shape.0,
            1.0, 1.0, None
        );
        
        // Bias gradients: sum over batch and spatial dimensions
        let dy_host = dy.to_host(&self.backend);
        let mut db_data = vec![0.0f32; self.out_channels];
        for i in 0..dy_host.len() {
            let channel = i % self.out_channels;
            db_data[channel] += dy_host[i];
        }
        self.db = Tensor::from_host(&self.backend, &db_data, (1, self.out_channels));
        
        // Input gradients: dX = dY * W^T (with proper reshaping)
        self.backend.gemm(
            &dy.buf, &self.w.buf, Arc::get_mut(&mut dx.buf).unwrap(),
            dx.shape.0, dx.shape.1, dy.shape.1,
            1.0, 0.0, None
        );
        
        dx
    }
    
    fn params(&self) -> Vec<&Tensor<B>> {
        vec![&self.w, &self.b]
    }
    
    fn grads(&mut self) -> Vec<&mut Tensor<B>> {
        vec![&mut self.dw, &mut self.db]
    }
    
    fn zero_grads(&mut self) {
        self.dw = Tensor::zeros(&self.backend, self.dw.shape.0, self.dw.shape.1);
        self.db = Tensor::zeros(&self.backend, self.db.shape.0, self.db.shape.1);
    }

    fn update_params(&mut self, lr: f32) {
        // Update parameters using simple SGD on the active backend
        self.w.sgd_update(&self.dw, &self.backend, lr);
        self.b.sgd_update(&self.db, &self.backend, lr);
    }
}

impl<B: Backend + Clone> Conv2d<B> { }

pub struct MaxPool2d<B: Backend> {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub last_input: Option<Tensor<B>>,
    pub last_indices: Option<Tensor<B>>, // Store argmax indices for backward pass (GPU-compatible)
    backend: B,
}

impl<B: Backend + Clone> MaxPool2d<B> {
    pub fn new(backend: B, kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride,
            last_input: None,
            last_indices: None,
            backend,
        }
    }
    
    fn compute_output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let (kernel_h, kernel_w) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        
        let output_h = (input_h - kernel_h) / stride_h + 1;
        let output_w = (input_w - kernel_w) / stride_w + 1;
        
        (output_h, output_w)
    }
}

impl<B: Backend + Clone> Layer<B> for MaxPool2d<B> {
    fn forward(&mut self, x: &Tensor<B>) -> Tensor<B> {
        self.last_input = Some(x.clone());
        
        // Assume input shape is (batch_size, channels * input_h * input_w)
        let batch_size = x.shape.0;
        let total_spatial = x.shape.1;
        
        // For MNIST: 1 channel, 28x28 -> assume we can infer dimensions
        // This is a simplified approach - proper implementation would track tensor shapes
        let channels = 1; // For MNIST after first conv layer, this might be different
        let spatial_per_channel = total_spatial / channels;
        let input_h = (spatial_per_channel as f32).sqrt() as usize;
        let input_w = input_h;
        
        let (output_h, output_w) = self.compute_output_size(input_h, input_w);
        
        // M6 Improvement: Native GPU MaxPool Operations
        let mut output = Tensor::zeros(&self.backend, batch_size, channels * output_h * output_w);
        let mut indices = Tensor::zeros(&self.backend, batch_size, channels * output_h * output_w);
        
        // Use native GPU maxpool2d kernel
        self.backend.maxpool2d(
            &x.buf,
            Arc::get_mut(&mut output.buf).unwrap(),
            Arc::get_mut(&mut indices.buf).unwrap(),
            (batch_size, channels, input_h, input_w),
            self.kernel_size,
            self.stride
        );
        
        self.last_indices = Some(indices);
        output
    }
    
    fn backward(&mut self, dy: &Tensor<B>) -> Tensor<B> {
        let x = self.last_input.as_ref().expect("Must call forward before backward");
        let indices = self.last_indices.as_ref().expect("Must call forward before backward");
        
        let _dx = Tensor::zeros(&self.backend, x.shape.0, x.shape.1);
        
        // Backward pass: scatter gradients to max locations
        // For now, use CPU implementation - can be optimized with a GPU scatter kernel later
        let dy_host = dy.to_host(&self.backend);
        let indices_host = indices.to_host(&self.backend);
        let mut dx_data = vec![0.0f32; x.len()];
        
        for (out_idx, &max_input_idx_f32) in indices_host.iter().enumerate() {
            let max_input_idx = max_input_idx_f32 as usize;
            if max_input_idx < dx_data.len() {
                dx_data[max_input_idx] += dy_host[out_idx];
            }
        }
        
        let dx_tensor = Tensor::from_host(&self.backend, &dx_data, (x.shape.0, x.shape.1));
        dx_tensor
    }
    
    fn params(&self) -> Vec<&Tensor<B>> {
        vec![]
    }
    
    fn grads(&mut self) -> Vec<&mut Tensor<B>> {
        vec![]
    }
    
    fn zero_grads(&mut self) {
        // MaxPool has no parameters, so no gradients to zero
    }

    fn update_params(&mut self, _lr: f32) {
        // MaxPool has no parameters
    }
}
