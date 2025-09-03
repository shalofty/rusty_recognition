use crate::{backend::Backend, tensor::Tensor, nn::{Layer, Linear, ReLU}};

pub struct MLP<B: Backend> {
    layers: Vec<Box<dyn Layer<B>>>,
}

impl<B: Backend + Clone + 'static> MLP<B> {
    pub fn new(backend: B, layer_sizes: &[usize]) -> Self {
        let mut layers: Vec<Box<dyn Layer<B>>> = Vec::new();
        
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Box::new(Linear::new(backend.clone(), layer_sizes[i], layer_sizes[i + 1])));
            if i < layer_sizes.len() - 2 {
                layers.push(Box::new(ReLU::new(backend.clone())));
            }
        }
        
        Self { layers }
    }
    
    pub fn forward(&mut self, x: &Tensor<B>) -> Tensor<B> {
        let mut output = x.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }
    
    pub fn backward(&mut self, dy: &Tensor<B>) -> Tensor<B> {
        let mut grad = dy.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
        grad
    }
    
    pub fn params(&self) -> Vec<&Tensor<B>> {
        self.layers.iter().flat_map(|layer| layer.params()).collect()
    }
    
    pub fn grads(&mut self) -> Vec<&mut Tensor<B>> {
        self.layers.iter_mut().flat_map(|layer| layer.grads()).collect()
    }
    
    pub fn zero_grads(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grads();
        }
    }
    
    pub fn update_params(&mut self, _lr: f32) {
        // Apply per-layer updates via the Layer trait
        for layer in &mut self.layers {
            layer.update_params(_lr);
        }
    }
}
