pub fn naive_gemm(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
    alpha: f32, beta: f32
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
}

pub fn add_bias(output: &mut [f32], bias: &[f32], m: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            output[i * n + j] += bias[j];
        }
    }
}