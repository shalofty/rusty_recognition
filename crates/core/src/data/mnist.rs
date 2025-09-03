use std::fs::File;
use std::io::{BufReader, Read};
use anyhow::{Result, Context};

pub struct MnistDataset {
    pub images: Vec<f32>,
    pub labels: Vec<u8>,
    pub num_samples: usize,
    pub image_size: (usize, usize),
}

impl MnistDataset {
    pub fn load_train() -> Result<Self> {
        Self::load("data/data/train-images-idx3-ubyte", "data/data/train-labels-idx1-ubyte")
    }
    
    pub fn load_test() -> Result<Self> {
        Self::load("data/data/t10k-images-idx3-ubyte", "data/data/t10k-labels-idx1-ubyte")
    }
    
    fn load(images_path: &str, labels_path: &str) -> Result<Self> {
        let images = Self::load_images(images_path)?;
        let labels = Self::load_labels(labels_path)?;
        
        let num_samples = labels.len();
        let image_size = (28, 28);
        
        Ok(Self {
            images,
            labels,
            num_samples,
            image_size,
        })
    }
    
    fn load_images(path: &str) -> Result<Vec<f32>> {
        let mut file = BufReader::new(File::open(path)
            .with_context(|| format!("Failed to open {}", path))?);
        
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        
        let mut num_images = [0u8; 4];
        file.read_exact(&mut num_images)?;
        let num_images = u32::from_be_bytes(num_images) as usize;
        
        let mut num_rows = [0u8; 4];
        file.read_exact(&mut num_rows)?;
        let num_rows = u32::from_be_bytes(num_rows) as usize;
        
        let mut num_cols = [0u8; 4];
        file.read_exact(&mut num_cols)?;
        let num_cols = u32::from_be_bytes(num_cols) as usize;
        
        let total_pixels = num_images * num_rows * num_cols;
        let mut pixels = vec![0u8; total_pixels];
        file.read_exact(&mut pixels)?;
        
        let normalized: Vec<f32> = pixels.into_iter()
            .map(|p| p as f32 / 255.0)
            .collect();
        
        Ok(normalized)
    }
    
    fn load_labels(path: &str) -> Result<Vec<u8>> {
        let mut file = BufReader::new(File::open(path)
            .with_context(|| format!("Failed to open {}", path))?);
        
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        
        let mut num_labels = [0u8; 4];
        file.read_exact(&mut num_labels)?;
        let num_labels = u32::from_be_bytes(num_labels) as usize;
        
        let mut labels = vec![0u8; num_labels];
        file.read_exact(&mut labels)?;
        
        Ok(labels)
    }
    
    pub fn get_batch(&self, batch_start: usize, batch_size: usize) -> (Vec<f32>, Vec<u8>) {
        let end = (batch_start + batch_size).min(self.num_samples);
        let actual_batch_size = end - batch_start;
        
        let pixels_per_image = self.image_size.0 * self.image_size.1;
        let mut batch_images = Vec::with_capacity(actual_batch_size * pixels_per_image);
        let mut batch_labels = Vec::with_capacity(actual_batch_size);
        
        for i in batch_start..end {
            let start_idx = i * pixels_per_image;
            let end_idx = start_idx + pixels_per_image;
            batch_images.extend_from_slice(&self.images[start_idx..end_idx]);
            batch_labels.push(self.labels[i]);
        }
        
        (batch_images, batch_labels)
    }
}