use metal::*;

#[derive(Clone)]
pub struct MetalBuffer {
    pub buffer: Buffer,
    pub length: usize,
}

impl MetalBuffer {
    pub fn new(device: &Device, length: usize) -> Self {
        let byte_size = length * std::mem::size_of::<f32>();
        let buffer = device.new_buffer(byte_size as u64, MTLResourceOptions::StorageModeShared);
        
        Self { buffer, length }
    }
    
    pub fn from_slice(device: &Device, data: &[f32]) -> Self {
        let byte_size = data.len() * std::mem::size_of::<f32>();
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            byte_size as u64,
            MTLResourceOptions::StorageModeShared
        );
        
        Self {
            buffer,
            length: data.len(),
        }
    }
    
    pub fn upload(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.length, "Data length must match buffer length");
        
        let contents = self.buffer.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                contents as *mut u8,
                data.len() * std::mem::size_of::<f32>(),
            );
        }
    }
    
    pub fn download(&self, output: &mut [f32]) {
        assert_eq!(output.len(), self.length, "Output length must match buffer length");
        
        let contents = self.buffer.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(
                contents as *const u8,
                output.as_mut_ptr() as *mut u8,
                output.len() * std::mem::size_of::<f32>(),
            );
        }
    }
    
    pub fn zero(&mut self) {
        let contents = self.buffer.contents();
        unsafe {
            std::ptr::write_bytes(
                contents as *mut u8,
                0,
                self.length * std::mem::size_of::<f32>(),
            );
        }
    }
}