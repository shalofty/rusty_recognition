pub trait Backend: Send + Sync + 'static {
    type Buf: Send + Sync + Clone;
    
    fn alloc(&self, len: usize) -> Self::Buf;
    fn upload(&self, host: &[f32]) -> Self::Buf;
    fn download(&self, buf: &Self::Buf, out: &mut [f32]);
    
    fn relu(&self, x: &Self::Buf, y: &mut Self::Buf, n: usize);
    fn gemm(&self, a: &Self::Buf, b: &Self::Buf, c: &mut Self::Buf,
            m: usize, n: usize, k: usize, alpha: f32, beta: f32, bias: Option<&Self::Buf>);
    fn softmax_xent(&self, logits: &Self::Buf, labels: &Self::Buf,
                    loss: &mut Self::Buf, dlogits: &mut Self::Buf,
                    batch: usize, classes: usize);
    fn drelu(&self, dy: &Self::Buf, x: &Self::Buf, dx: &mut Self::Buf, n: usize);
    fn sgd_step(&self, params: &mut Self::Buf, grads: &Self::Buf, lr: f32, momentum: Option<f32>, 
                velocity: Option<&mut Self::Buf>, n: usize);
    
    // M6 Improvements: Im2Col integration for proper convolution
    fn im2col(&self, input: &Self::Buf, output: &mut Self::Buf,
              input_dims: (usize, usize, usize, usize), // N, C, H, W
              kernel_size: (usize, usize),
              padding: (usize, usize),
              stride: (usize, usize),
              output_dims: (usize, usize)); // output_h, output_w
    
    // M6 Improvements: Native GPU MaxPool operations
    fn maxpool2d(&self, input: &Self::Buf, output: &mut Self::Buf, indices: &mut Self::Buf,
                 input_dims: (usize, usize, usize, usize),
                 kernel_size: (usize, usize),
                 stride: (usize, usize));
    
    // M6 Advanced Optimization: Fused Linear + ReLU operation
    fn linear_relu_fused(&self, x: &Self::Buf, w: &Self::Buf, b: &Self::Buf, y: &mut Self::Buf,
                        batch_size: usize, in_features: usize, out_features: usize);
}

#[derive(Clone, Copy)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Buf = Vec<f32>;
    
    fn alloc(&self, len: usize) -> Self::Buf {
        vec![0.0; len]
    }
    
    fn upload(&self, host: &[f32]) -> Self::Buf {
        host.to_vec()
    }
    
    fn download(&self, buf: &Self::Buf, out: &mut [f32]) {
        out.copy_from_slice(buf);
    }
    
    fn relu(&self, x: &Self::Buf, y: &mut Self::Buf, n: usize) {
        for i in 0..n {
            y[i] = x[i].max(0.0);
        }
    }
    
    fn gemm(&self, a: &Self::Buf, b: &Self::Buf, c: &mut Self::Buf,
            m: usize, n: usize, k: usize, alpha: f32, beta: f32, bias: Option<&Self::Buf>) {
        // C = alpha * A * B + beta * C
        // A is m x k, B is k x n, C is m x n
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = alpha * sum + beta * c[i * n + j];
                if let Some(bias_buf) = bias {
                    c[i * n + j] += bias_buf[j];
                }
            }
        }
    }
    
    fn softmax_xent(&self, logits: &Self::Buf, labels: &Self::Buf,
                    loss: &mut Self::Buf, dlogits: &mut Self::Buf,
                    batch: usize, classes: usize) {
        let mut total_loss = 0.0;
        
        for b in 0..batch {
            let offset = b * classes;
            let label = labels[b] as usize;
            
            // Find max for numerical stability
            let max_logit = logits[offset..offset + classes].iter().copied()
                .fold(f32::NEG_INFINITY, f32::max);
            
            // Compute exp and sum
            let mut sum_exp = 0.0;
            for c in 0..classes {
                let exp_val = (logits[offset + c] - max_logit).exp();
                dlogits[offset + c] = exp_val;
                sum_exp += exp_val;
            }
            
            // Compute probabilities and loss
            let prob_true_class = dlogits[offset + label] / sum_exp;
            total_loss -= (prob_true_class + 1e-7).ln();
            
            // Compute gradients: prob - target
            for c in 0..classes {
                let prob = dlogits[offset + c] / sum_exp;
                dlogits[offset + c] = (prob - if c == label { 1.0 } else { 0.0 }) / (batch as f32);
            }
        }
        
        loss[0] = total_loss / batch as f32;
    }
    
    fn drelu(&self, dy: &Self::Buf, x: &Self::Buf, dx: &mut Self::Buf, n: usize) {
        for i in 0..n {
            dx[i] = if x[i] > 0.0 { dy[i] } else { 0.0 };
        }
    }
    
    fn sgd_step(&self, params: &mut Self::Buf, grads: &Self::Buf, lr: f32, momentum: Option<f32>,
                velocity: Option<&mut Self::Buf>, n: usize) {
        match (momentum, velocity) {
            (Some(mom), Some(vel)) => {
                for i in 0..n {
                    vel[i] = mom * vel[i] + lr * grads[i];
                    params[i] -= vel[i];
                }
            }
            _ => {
                for i in 0..n {
                    params[i] -= lr * grads[i];
                }
            }
        }
    }
    
    fn im2col(&self, input: &Self::Buf, output: &mut Self::Buf,
              input_dims: (usize, usize, usize, usize), // N, C, H, W
              kernel_size: (usize, usize),
              padding: (usize, usize),
              stride: (usize, usize),
              output_dims: (usize, usize)) {
        let (n, c, h, w) = input_dims;
        let (kernel_h, kernel_w) = kernel_size;
        let (pad_h, pad_w) = padding;
        let (stride_h, stride_w) = stride;
        let (output_h, output_w) = output_dims;
        
        for batch_idx in 0..n {
            for out_y in 0..output_h {
                for out_x in 0..output_w {
                    for c_idx in 0..c {
                        for kernel_y in 0..kernel_h {
                            for kernel_x in 0..kernel_w {
                                let in_y = (out_y * stride_h) as i32 - pad_h as i32 + kernel_y as i32;
                                let in_x = (out_x * stride_w) as i32 - pad_w as i32 + kernel_x as i32;
                                
                                let col_idx = batch_idx * (output_h * output_w) + out_y * output_w + out_x;
                                let row_idx = c_idx * (kernel_h * kernel_w) + kernel_y * kernel_w + kernel_x;
                                let output_idx = row_idx * (output_h * output_w * n) + col_idx;
                                
                                if in_y >= 0 && in_y < h as i32 && in_x >= 0 && in_x < w as i32 {
                                    let input_idx = batch_idx * (c * h * w) + c_idx * (h * w) + 
                                                  (in_y as usize) * w + (in_x as usize);
                                    output[output_idx] = input[input_idx];
                                } else {
                                    output[output_idx] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    fn maxpool2d(&self, input: &Self::Buf, output: &mut Self::Buf, indices: &mut Self::Buf,
                 input_dims: (usize, usize, usize, usize),
                 kernel_size: (usize, usize),
                 stride: (usize, usize)) {
        let (n, c, h, w) = input_dims;
        let (kernel_h, kernel_w) = kernel_size;
        let (stride_h, stride_w) = stride;
        
        let output_h = (h - kernel_h) / stride_h + 1;
        let output_w = (w - kernel_w) / stride_w + 1;
        
        for batch_idx in 0..n {
            for c_idx in 0..c {
                for out_y in 0..output_h {
                    for out_x in 0..output_w {
                        let start_y = out_y * stride_h;
                        let start_x = out_x * stride_w;
                        
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;
                        
                        for ky in 0..kernel_h {
                            for kx in 0..kernel_w {
                                let in_y = start_y + ky;
                                let in_x = start_x + kx;
                                
                                if in_y < h && in_x < w {
                                    let input_idx = batch_idx * (c * h * w) + c_idx * (h * w) + 
                                                  in_y * w + in_x;
                                    if input[input_idx] > max_val {
                                        max_val = input[input_idx];
                                        max_idx = input_idx;
                                    }
                                }
                            }
                        }
                        
                        let output_idx = batch_idx * (c * output_h * output_w) + 
                                       c_idx * (output_h * output_w) + out_y * output_w + out_x;
                        output[output_idx] = max_val;
                        // Note: CPU backend stores indices as f32 in the same buffer type
                        indices[output_idx] = max_idx as f32;
                    }
                }
            }
        }
    }
    
    fn linear_relu_fused(&self, x: &Self::Buf, w: &Self::Buf, b: &Self::Buf, y: &mut Self::Buf,
                        batch_size: usize, in_features: usize, out_features: usize) {
        // Fused Linear + ReLU: y = max(0, x * w + b)
        for i in 0..batch_size {
            for j in 0..out_features {
                let mut sum = b[j];
                for k in 0..in_features {
                    sum += x[i * in_features + k] * w[k * out_features + j];
                }
                y[i * out_features + j] = sum.max(0.0);
            }
        }
    }
}
