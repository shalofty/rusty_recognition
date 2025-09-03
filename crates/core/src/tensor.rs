use crate::backend::Backend;
use std::sync::Arc;

#[derive(Clone)]
pub struct Tensor<B: Backend> {
    pub buf: Arc<B::Buf>,
    pub shape: (usize, usize),
}

impl<B: Backend> Tensor<B> {
    pub fn zeros(backend: &B, rows: usize, cols: usize) -> Self {
        let len = rows * cols;
        let buf = backend.alloc(len);
        Self {
            buf: Arc::new(buf),
            shape: (rows, cols),
        }
    }
    
    pub fn from_host(backend: &B, data: &[f32], shape: (usize, usize)) -> Self {
        let buf = backend.upload(data);
        Self {
            buf: Arc::new(buf),
            shape,
        }
    }
    
    pub fn to_host(&self, backend: &B) -> Vec<f32> {
        let mut host_data = vec![0.0; self.shape.0 * self.shape.1];
        backend.download(&self.buf, &mut host_data);
        host_data
    }
    
    pub fn len(&self) -> usize {
        self.shape.0 * self.shape.1
    }
    
    // M6 Improvement: Method to update parameters in-place using SGD
    pub fn sgd_update(&mut self, grads: &Tensor<B>, backend: &B, lr: f32) {
        // Create a new buffer for the updated parameters
        let current_data = self.to_host(backend);
        let grad_data = grads.to_host(backend);
        
        let mut updated_data = current_data.clone();
        for i in 0..updated_data.len() {
            updated_data[i] -= lr * grad_data[i];
        }
        
        // Replace the buffer with the updated one
        self.buf = Arc::new(backend.upload(&updated_data));
    }
}