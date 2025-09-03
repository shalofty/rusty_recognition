use crate::{backend::Backend, tensor::Tensor};

pub trait Optimizer<B: Backend> {
    fn step(&mut self, params: &mut [&mut Tensor<B>], grads: &[&Tensor<B>]);
}

pub struct Sgd<B: Backend> {
    lr: f32,
    momentum: Option<f32>,
    velocities: Vec<Option<Tensor<B>>>,
    backend: B,
}

impl<B: Backend + Clone> Sgd<B> {
    pub fn new(backend: B, lr: f32, momentum: Option<f32>) -> Self {
        Self {
            lr,
            momentum,
            velocities: Vec::new(),
            backend,
        }
    }
}

impl<B: Backend + Clone> Optimizer<B> for Sgd<B> {
    fn step(&mut self, params: &mut [&mut Tensor<B>], grads: &[&Tensor<B>]) {
        if self.velocities.is_empty() {
            self.velocities.resize_with(params.len(), || {
                if self.momentum.is_some() {
                    Some(Tensor::zeros(&self.backend, 1, 1))
                } else {
                    None
                }
            });
        }
        
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let velocity = self.velocities.get_mut(i).and_then(|v| v.as_mut());
            let param_len = param.len();
            
            if let Some(velocity_buf) = velocity.map(|v| &mut v.buf) {
                self.backend.sgd_step(
                    std::sync::Arc::get_mut(&mut param.buf).unwrap(),
                    &grad.buf,
                    self.lr,
                    self.momentum,
                    Some(std::sync::Arc::get_mut(velocity_buf).unwrap()),
                    param_len,
                );
            } else {
                self.backend.sgd_step(
                    std::sync::Arc::get_mut(&mut param.buf).unwrap(),
                    &grad.buf,
                    self.lr,
                    None,
                    None,
                    param_len,
                );
            }
        }
    }
}