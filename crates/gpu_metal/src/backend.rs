use mnist_core::backend::Backend;
use crate::{MetalDevice, MetalBuffer, LaunchConfig, dispatch_kernel};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct MetalBackend {
    device: Arc<Mutex<MetalDevice>>,
}

impl MetalBackend {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            device: Arc::new(Mutex::new(MetalDevice::new()?)),
        })
    }
}

impl Backend for MetalBackend {
    type Buf = MetalBuffer;
    
    fn alloc(&self, len: usize) -> Self::Buf {
        let device = self.device.lock().unwrap();
        MetalBuffer::new(&device.device, len)
    }
    
    fn upload(&self, host: &[f32]) -> Self::Buf {
        let device = self.device.lock().unwrap();
        MetalBuffer::from_slice(&device.device, host)
    }
    
    fn download(&self, buf: &Self::Buf, out: &mut [f32]) {
        buf.download(out);
    }
    
    fn relu(&self, x: &Self::Buf, y: &mut Self::Buf, n: usize) {
        let mut device = self.device.lock().unwrap();
        let pipeline = device.get_or_create_pipeline("relu_forward").unwrap().clone();
        let command_buffer = device.new_command_buffer();
        drop(device); // Release the lock
        
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_buffer(0, Some(&x.buffer), 0);
        encoder.set_buffer(1, Some(&y.buffer), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &(n as u32) as *const u32 as *const _);
        
        let config = LaunchConfig::new_1d(n, 256);
        dispatch_kernel(&encoder, &pipeline, &config);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    
    fn gemm(&self, a: &Self::Buf, b: &Self::Buf, c: &mut Self::Buf,
            m: usize, n: usize, k: usize, alpha: f32, beta: f32, bias: Option<&Self::Buf>) {
        let mut device = self.device.lock().unwrap();
        let pipeline = device.get_or_create_pipeline("naive_gemm").unwrap().clone();
        let command_buffer = device.new_command_buffer();
        drop(device);
        
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_buffer(0, Some(&a.buffer), 0);
        encoder.set_buffer(1, Some(&b.buffer), 0);
        encoder.set_buffer(2, Some(&c.buffer), 0);
        
        let params = [m as u32, n as u32, k as u32];
        encoder.set_bytes(3, std::mem::size_of_val(&params) as u64, params.as_ptr() as *const _);
        encoder.set_bytes(4, std::mem::size_of::<f32>() as u64, &alpha as *const f32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<f32>() as u64, &beta as *const f32 as *const _);
        
        if let Some(bias_buf) = bias {
            encoder.set_buffer(6, Some(&bias_buf.buffer), 0);
        }
        
        // Use tiled kernel for larger matrices
        let config = if m >= 64 && n >= 64 {
            LaunchConfig::new_2d((m + 15) / 16, (n + 15) / 16, 16, 16)
        } else {
            LaunchConfig::new_2d(m, n, 16, 16)
        };
        dispatch_kernel(&encoder, &pipeline, &config);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    
    fn softmax_xent(&self, logits: &Self::Buf, labels: &Self::Buf,
                    loss: &mut Self::Buf, dlogits: &mut Self::Buf,
                    batch: usize, classes: usize) {
        let mut device = self.device.lock().unwrap();
        let pipeline = device.get_or_create_pipeline("softmax_xent").unwrap().clone();
        let command_buffer = device.new_command_buffer();
        drop(device);
        
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_buffer(0, Some(&logits.buffer), 0);
        encoder.set_buffer(1, Some(&labels.buffer), 0);
        encoder.set_buffer(2, Some(&loss.buffer), 0);
        encoder.set_buffer(3, Some(&dlogits.buffer), 0);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(batch as u32) as *const u32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &(classes as u32) as *const u32 as *const _);
        
        let config = LaunchConfig::new_1d(batch, 256);
        dispatch_kernel(&encoder, &pipeline, &config);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Sum the per-sample losses on CPU for now (could optimize with GPU reduction later)
        let mut loss_data = vec![0.0f32; loss.length];
        loss.download(&mut loss_data);
        let total_loss: f32 = loss_data.iter().sum();
        let avg_loss = total_loss / batch as f32;
        
        // Write back the average loss to first element
        loss_data[0] = avg_loss;
        loss.upload(&loss_data);
    }
    
    fn drelu(&self, dy: &Self::Buf, x: &Self::Buf, dx: &mut Self::Buf, n: usize) {
        let mut device = self.device.lock().unwrap();
        let pipeline = device.get_or_create_pipeline("drelu").unwrap().clone();
        let command_buffer = device.new_command_buffer();
        drop(device);
        
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_buffer(0, Some(&dy.buffer), 0);
        encoder.set_buffer(1, Some(&x.buffer), 0);
        encoder.set_buffer(2, Some(&dx.buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(n as u32) as *const u32 as *const _);
        
        let config = LaunchConfig::new_1d(n, 256);
        dispatch_kernel(&encoder, &pipeline, &config);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    
    fn sgd_step(&self, params: &mut Self::Buf, grads: &Self::Buf, lr: f32, momentum: Option<f32>,
                velocity: Option<&mut Self::Buf>, n: usize) {
        let mut device = self.device.lock().unwrap();
        let pipeline = device.get_or_create_pipeline("sgd_step").unwrap().clone();
        let command_buffer = device.new_command_buffer();
        drop(device);
        
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_buffer(0, Some(&params.buffer), 0);
        encoder.set_buffer(1, Some(&grads.buffer), 0);
        encoder.set_bytes(2, std::mem::size_of::<f32>() as u64, &lr as *const f32 as *const _);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(n as u32) as *const u32 as *const _);
        
        if let Some(vel_buf) = velocity {
            encoder.set_buffer(4, Some(&vel_buf.buffer), 0);
            let mom = momentum.unwrap_or(0.0);
            encoder.set_bytes(5, std::mem::size_of::<f32>() as u64, &mom as *const f32 as *const _);
        }
        
        let config = LaunchConfig::new_1d(n, 256);
        dispatch_kernel(&encoder, &pipeline, &config);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    
    fn im2col(&self, input: &Self::Buf, output: &mut Self::Buf,
              input_dims: (usize, usize, usize, usize), // N, C, H, W
              kernel_size: (usize, usize),
              padding: (usize, usize),
              stride: (usize, usize),
              output_dims: (usize, usize)) {
        let mut device = self.device.lock().unwrap();
        let pipeline = device.get_or_create_pipeline("im2col").unwrap().clone();
        let command_buffer = device.new_command_buffer();
        drop(device);
        
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        
        let (n, c, h, w) = input_dims;
        let input_dims_arr = [n as u32, c as u32, h as u32, w as u32];
        encoder.set_bytes(2, std::mem::size_of_val(&input_dims_arr) as u64, 
                         input_dims_arr.as_ptr() as *const _);
        
        let kernel_size_arr = [kernel_size.0 as u32, kernel_size.1 as u32];
        encoder.set_bytes(3, std::mem::size_of_val(&kernel_size_arr) as u64,
                         kernel_size_arr.as_ptr() as *const _);
        
        let padding_arr = [padding.0 as u32, padding.1 as u32];
        encoder.set_bytes(4, std::mem::size_of_val(&padding_arr) as u64,
                         padding_arr.as_ptr() as *const _);
        
        let stride_arr = [stride.0 as u32, stride.1 as u32];
        encoder.set_bytes(5, std::mem::size_of_val(&stride_arr) as u64,
                         stride_arr.as_ptr() as *const _);
        
        let (output_h, output_w) = output_dims;
        let output_dims_arr = [output_h as u32, output_w as u32];
        encoder.set_bytes(6, std::mem::size_of_val(&output_dims_arr) as u64,
                         output_dims_arr.as_ptr() as *const _);
        
        let total_cols = output_h * output_w * n;
        let total_rows = kernel_size.0 * kernel_size.1 * c;
        let config = LaunchConfig::new_2d(total_cols, total_rows, 16, 16);
        dispatch_kernel(&encoder, &pipeline, &config);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    
    fn maxpool2d(&self, input: &Self::Buf, output: &mut Self::Buf, indices: &mut Self::Buf,
                 input_dims: (usize, usize, usize, usize),
                 kernel_size: (usize, usize),
                 stride: (usize, usize)) {
        let mut device = self.device.lock().unwrap();
        let pipeline = device.get_or_create_pipeline("maxpool2d_forward").unwrap().clone();
        let command_buffer = device.new_command_buffer();
        drop(device);
        
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        encoder.set_buffer(2, Some(&indices.buffer), 0);
        
        let (n, c, h, w) = input_dims;
        let input_dims_arr = [n as u32, c as u32, h as u32, w as u32];
        encoder.set_bytes(3, std::mem::size_of_val(&input_dims_arr) as u64,
                         input_dims_arr.as_ptr() as *const _);
        
        let kernel_size_arr = [kernel_size.0 as u32, kernel_size.1 as u32];
        encoder.set_bytes(4, std::mem::size_of_val(&kernel_size_arr) as u64,
                         kernel_size_arr.as_ptr() as *const _);
        
        let stride_arr = [stride.0 as u32, stride.1 as u32];
        encoder.set_bytes(5, std::mem::size_of_val(&stride_arr) as u64,
                         stride_arr.as_ptr() as *const _);
        
        let output_h = (h - kernel_size.0) / stride.0 + 1;
        let output_w = (w - kernel_size.1) / stride.1 + 1;
        let output_dims_arr = [output_h as u32, output_w as u32];
        encoder.set_bytes(6, std::mem::size_of_val(&output_dims_arr) as u64,
                         output_dims_arr.as_ptr() as *const _);
        
        let config = LaunchConfig::new_3d(output_h * output_w, c, n, 8, 8, 4);
        dispatch_kernel(&encoder, &pipeline, &config);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    
    fn linear_relu_fused(&self, x: &Self::Buf, w: &Self::Buf, b: &Self::Buf, y: &mut Self::Buf,
                        batch_size: usize, in_features: usize, out_features: usize) {
        let mut device = self.device.lock().unwrap();
        let pipeline = device.get_or_create_pipeline("linear_relu_forward").unwrap().clone();
        let command_buffer = device.new_command_buffer();
        drop(device);
        
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_buffer(0, Some(&x.buffer), 0);
        encoder.set_buffer(1, Some(&w.buffer), 0);
        encoder.set_buffer(2, Some(&b.buffer), 0);
        encoder.set_buffer(3, Some(&y.buffer), 0);
        
        let dims = [batch_size as u32, in_features as u32, out_features as u32];
        encoder.set_bytes(4, std::mem::size_of_val(&dims) as u64, dims.as_ptr() as *const _);
        
        let config = LaunchConfig::new_2d(out_features, batch_size, 16, 16);
        dispatch_kernel(&encoder, &pipeline, &config);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}