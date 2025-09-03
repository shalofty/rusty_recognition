pub fn softmax_forward(logits: &[f32], output: &mut [f32], batch_size: usize, num_classes: usize) {
    for b in 0..batch_size {
        let offset = b * num_classes;
        let logits_batch = &logits[offset..offset + num_classes];
        let output_batch = &mut output[offset..offset + num_classes];
        
        let max_logit = logits_batch.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        let mut sum_exp = 0.0;
        for (i, &logit) in logits_batch.iter().enumerate() {
            let exp_val = (logit - max_logit).exp();
            output_batch[i] = exp_val;
            sum_exp += exp_val;
        }
        
        for prob in output_batch.iter_mut() {
            *prob /= sum_exp;
        }
    }
}

pub fn cross_entropy_loss(probs: &[f32], labels: &[usize], batch_size: usize, num_classes: usize) -> f32 {
    let mut total_loss = 0.0;
    
    for b in 0..batch_size {
        let offset = b * num_classes;
        let label = labels[b];
        total_loss -= probs[offset + label].ln();
    }
    
    total_loss / batch_size as f32
}

pub fn softmax_cross_entropy_backward(
    probs: &[f32], labels: &[usize], grad_logits: &mut [f32], 
    batch_size: usize, num_classes: usize
) {
    for b in 0..batch_size {
        let offset = b * num_classes;
        let label = labels[b];
        
        for c in 0..num_classes {
            grad_logits[offset + c] = probs[offset + c] - if c == label { 1.0 } else { 0.0 };
        }
    }
    
    let scale = 1.0 / batch_size as f32;
    for grad in grad_logits.iter_mut() {
        *grad *= scale;
    }
}