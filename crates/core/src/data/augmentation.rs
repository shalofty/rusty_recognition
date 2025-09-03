use rand::{thread_rng, Rng};

// M6 Improvement: Data augmentation for better accuracy
pub struct MnistAugmentedDataset {
    rotation_range: f32,      // Maximum rotation in radians
    shift_range: f32,         // Maximum shift as fraction of image size  
    noise_std: f32,          // Standard deviation of Gaussian noise
}

impl MnistAugmentedDataset {
    pub fn new(rotation_range: f32, shift_range: f32, noise_std: f32) -> Self {
        Self {
            rotation_range,
            shift_range, 
            noise_std,
        }
    }
    
    pub fn augment_batch(&self, images: &mut [f32]) {
        let image_size = 28;
        let num_images = images.len() / (image_size * image_size);
        
        for img_idx in 0..num_images {
            let start = img_idx * image_size * image_size;
            let end = start + image_size * image_size;
            let image = &mut images[start..end];
            
            // Apply random augmentations
            if self.rotation_range > 0.0 {
                self.random_rotation(image, image_size);
            }
            if self.shift_range > 0.0 {
                self.random_shift(image, image_size);
            }
            if self.noise_std > 0.0 {
                self.random_noise(image);
            }
        }
    }
    
    fn random_rotation(&self, image: &mut [f32], size: usize) {
        // Simple rotation by small angles
        let angle = thread_rng().gen_range(-self.rotation_range..self.rotation_range);
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        let center = size as f32 / 2.0;
        
        let mut rotated = vec![0.0f32; size * size];
        
        for y in 0..size {
            for x in 0..size {
                // Rotate around center
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                
                let src_x = center + cos_angle * dx - sin_angle * dy;
                let src_y = center + sin_angle * dx + cos_angle * dy;
                
                // Bilinear interpolation
                if src_x >= 0.0 && src_x < size as f32 && src_y >= 0.0 && src_y < size as f32 {
                    let x1 = src_x.floor() as usize;
                    let y1 = src_y.floor() as usize;
                    let x2 = (x1 + 1).min(size - 1);
                    let y2 = (y1 + 1).min(size - 1);
                    
                    let fx = src_x - x1 as f32;
                    let fy = src_y - y1 as f32;
                    
                    let val = (1.0 - fx) * (1.0 - fy) * image[y1 * size + x1] +
                              fx * (1.0 - fy) * image[y1 * size + x2] +
                              (1.0 - fx) * fy * image[y2 * size + x1] +
                              fx * fy * image[y2 * size + x2];
                    
                    rotated[y * size + x] = val;
                }
            }
        }
        
        image.copy_from_slice(&rotated);
    }
    
    fn random_shift(&self, image: &mut [f32], size: usize) {
        // Random translation
        let shift_pixels = (self.shift_range * size as f32) as i32;
        let dx = thread_rng().gen_range(-shift_pixels..=shift_pixels);
        let dy = thread_rng().gen_range(-shift_pixels..=shift_pixels);
        
        let mut shifted = vec![0.0f32; size * size];
        
        for y in 0..size {
            for x in 0..size {
                let src_x = x as i32 - dx;
                let src_y = y as i32 - dy;
                
                if src_x >= 0 && src_x < size as i32 && src_y >= 0 && src_y < size as i32 {
                    shifted[y * size + x] = image[src_y as usize * size + src_x as usize];
                }
            }
        }
        
        image.copy_from_slice(&shifted);
    }
    
    fn random_noise(&self, image: &mut [f32]) {
        // Add Gaussian noise
        for pixel in image.iter_mut() {
            // Box-Muller transform for normal distribution
            let u1: f32 = thread_rng().gen();
            let u2: f32 = thread_rng().gen();
            let noise = self.noise_std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            
            *pixel = (*pixel + noise).clamp(0.0, 1.0);
        }
    }
}

// Learning rate scheduler for better convergence
pub struct LRScheduler {
    initial_lr: f32,
    decay_factor: f32,
    decay_epochs: Vec<usize>,
}

impl LRScheduler {
    pub fn new(initial_lr: f32, decay_factor: f32, decay_epochs: Vec<usize>) -> Self {
        Self {
            initial_lr,
            decay_factor,
            decay_epochs,
        }
    }
    
    pub fn get_lr(&self, epoch: usize) -> f32 {
        let decay_count = self.decay_epochs.iter().filter(|&&e| e <= epoch).count();
        self.initial_lr * self.decay_factor.powi(decay_count as i32)
    }
}