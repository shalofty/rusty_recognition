use crate::{backend::Backend, tensor::Tensor, nn::{Layer, Conv2d, MaxPool2d, Linear, ReLU}, train::TrainableModel};

pub struct LeNet<B: Backend> {
    conv1: Conv2d<B>,
    pool1: MaxPool2d<B>,
    conv2: Conv2d<B>,
    pool2: MaxPool2d<B>,
    fc1: Linear<B>,
    relu1: ReLU<B>,
    fc2: Linear<B>,
    relu2: ReLU<B>,
    fc3: Linear<B>,
}

impl<B: Backend + Clone + 'static> LeNet<B> {
    pub fn new(backend: B) -> Self {
        Self {
            // Layer 1: Conv 5x5, 1 -> 6 channels, then 2x2 max pool
            conv1: Conv2d::new(backend.clone(), 1, 6, (5, 5), (2, 2), (1, 1)),
            pool1: MaxPool2d::new(backend.clone(), (2, 2), (2, 2)),
            
            // Layer 2: Conv 5x5, 6 -> 16 channels, then 2x2 max pool  
            conv2: Conv2d::new(backend.clone(), 6, 16, (5, 5), (0, 0), (1, 1)),
            pool2: MaxPool2d::new(backend.clone(), (2, 2), (2, 2)),
            
            // Fully connected layers: 16*5*5 -> 120 -> 84 -> 10
            fc1: Linear::new(backend.clone(), 16 * 5 * 5, 120),
            relu1: ReLU::new(backend.clone()),
            fc2: Linear::new(backend.clone(), 120, 84),
            relu2: ReLU::new(backend.clone()),
            fc3: Linear::new(backend.clone(), 84, 10),
        }
    }
    
    // M6 Improvement: He initialization version for better convergence
    pub fn new_he_initialization(backend: B) -> Self {
        Self {
            // Layer 1: Conv 5x5, 1 -> 6 channels, then 2x2 max pool
            conv1: Conv2d::new_he_initialization(backend.clone(), 1, 6, (5, 5), (2, 2), (1, 1)),
            pool1: MaxPool2d::new(backend.clone(), (2, 2), (2, 2)),
            
            // Layer 2: Conv 5x5, 6 -> 16 channels, then 2x2 max pool  
            conv2: Conv2d::new_he_initialization(backend.clone(), 6, 16, (5, 5), (0, 0), (1, 1)),
            pool2: MaxPool2d::new(backend.clone(), (2, 2), (2, 2)),
            
            // Fully connected layers: 16*5*5 -> 120 -> 84 -> 10
            fc1: Linear::new(backend.clone(), 16 * 5 * 5, 120),
            relu1: ReLU::new(backend.clone()),
            fc2: Linear::new(backend.clone(), 120, 84),
            relu2: ReLU::new(backend.clone()),
            fc3: Linear::new(backend.clone(), 84, 10),
        }
    }
    
    pub fn forward(&mut self, x: &Tensor<B>) -> Tensor<B> {
        // Input: (batch_size, 28*28) for MNIST
        // Need to treat as (batch_size, 1, 28, 28) conceptually
        
        let x1 = self.conv1.forward(x);      // (batch, 1, 28, 28) -> (batch, 6, 26, 26)
        let x2 = self.pool1.forward(&x1);    // (batch, 6, 26, 26) -> (batch, 6, 13, 13)
        let x3 = self.conv2.forward(&x2);    // (batch, 6, 13, 13) -> (batch, 16, 9, 9)
        let x4 = self.pool2.forward(&x3);    // (batch, 16, 9, 9) -> (batch, 16, 4, 4)
        
        // Flatten for FC layers: (batch, 16*4*4) = (batch, 256) 
        // Note: actual size will be 16*5*5 = 400 based on MNIST dimensions
        let x5 = self.fc1.forward(&x4);      // (batch, 400) -> (batch, 120)
        let x6 = self.relu1.forward(&x5);    // (batch, 120) -> (batch, 120)
        let x7 = self.fc2.forward(&x6);      // (batch, 120) -> (batch, 84)
        let x8 = self.relu2.forward(&x7);    // (batch, 84) -> (batch, 84)
        let logits = self.fc3.forward(&x8);  // (batch, 84) -> (batch, 10)
        
        logits
    }
    
    pub fn backward(&mut self, dy: &Tensor<B>) -> Tensor<B> {
        // Backward pass through all layers in reverse order
        let grad = self.fc3.backward(dy);
        let grad = self.relu2.backward(&grad);
        let grad = self.fc2.backward(&grad);
        let grad = self.relu1.backward(&grad);
        let grad = self.fc1.backward(&grad);
        let grad = self.pool2.backward(&grad);
        let grad = self.conv2.backward(&grad);
        let grad = self.pool1.backward(&grad);
        let grad = self.conv1.backward(&grad);
        
        grad
    }
    
    pub fn params(&self) -> Vec<&Tensor<B>> {
        let mut params = Vec::new();
        params.extend(self.conv1.params());
        params.extend(self.conv2.params());
        params.extend(self.fc1.params());
        params.extend(self.fc2.params());
        params.extend(self.fc3.params());
        params
    }
    
    pub fn grads(&mut self) -> Vec<&mut Tensor<B>> {
        let mut grads = Vec::new();
        grads.extend(self.conv1.grads());
        grads.extend(self.conv2.grads());
        grads.extend(self.fc1.grads());
        grads.extend(self.fc2.grads());
        grads.extend(self.fc3.grads());
        grads
    }
    
    pub fn zero_grads(&mut self) {
        self.conv1.zero_grads();
        self.conv2.zero_grads();
        self.fc1.zero_grads();
        self.fc2.zero_grads();
        self.fc3.zero_grads();
    }
    
    pub fn update_params(&mut self, lr: f32) {
        // M6 Improvement: Proper GPU parameter updates using SGD kernel
        // Update all layer parameters using the layer-specific update methods
        
        self.conv1.update_params(lr);
        self.conv2.update_params(lr);
        self.fc1.update_params(lr);
        self.fc2.update_params(lr);
        self.fc3.update_params(lr);
    }
}

// M6 Improvement: Implement TrainableModel trait for integration with optimized training loop
impl<B: Backend + Clone + 'static> TrainableModel<B> for LeNet<B> {
    fn forward(&mut self, x: &Tensor<B>) -> Tensor<B> {
        self.forward(x)
    }
    
    fn backward_pass(&mut self, logits: &Tensor<B>) {
        // This is a simplified backward pass - in a full implementation,
        // we would need to pass gradients from the loss back through the network
        // For now, this serves as a placeholder for the backward computation
        let _ = self.backward(logits);
    }
    
    fn update_params(&mut self, lr: f32) {
        self.update_params(lr)
    }
    
    fn zero_grads(&mut self) {
        self.zero_grads()
    }
}