pub fn relu_forward(input: &[f32], output: &mut [f32]) {
    for (i, &x) in input.iter().enumerate() {
        output[i] = x.max(0.0);
    }
}

pub fn relu_backward(grad_output: &[f32], input: &[f32], grad_input: &mut [f32]) {
    for i in 0..input.len() {
        grad_input[i] = if input[i] > 0.0 { grad_output[i] } else { 0.0 };
    }
}