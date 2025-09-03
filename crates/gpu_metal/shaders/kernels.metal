#include <metal_stdlib>
using namespace metal;

kernel void relu_forward(
    device const float* x   [[buffer(0)]],
    device float*       y   [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        float v = x[gid];
        y[gid] = v > 0.0f ? v : 0.0f;
    }
}

kernel void drelu(
    device const float* dy  [[buffer(0)]],
    device const float* x   [[buffer(1)]],
    device float*       dx  [[buffer(2)]],
    constant uint&      n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        dx[gid] = x[gid] > 0.0f ? dy[gid] : 0.0f;
    }
}

kernel void naive_gemm(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device float*       c      [[buffer(2)]],
    constant uint3&     dims   [[buffer(3)]], // m, n, k
    constant float&     alpha  [[buffer(4)]],
    constant float&     beta   [[buffer(5)]],
    device const float* bias   [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = dims.x;
    uint n = dims.y;
    uint k = dims.z;
    
    uint i = gid.y;
    uint j = gid.x;
    
    if (i < m && j < n) {
        float sum = 0.0f;
        for (uint p = 0; p < k; p++) {
            sum += a[i * k + p] * b[p * n + j];
        }
        
        c[i * n + j] = alpha * sum + beta * c[i * n + j];
        
        if (bias != nullptr) {
            c[i * n + j] += bias[j];
        }
    }
}

constant uint TILE_SIZE = 16;

kernel void matmul_tiled(
    device const float* a     [[buffer(0)]],
    device const float* b     [[buffer(1)]],
    device float*       c     [[buffer(2)]],
    constant uint3&     dims  [[buffer(3)]], // m, n, k
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    uint m = dims.x;
    uint n = dims.y;
    uint k = dims.z;
    
    uint row = group_id.y * TILE_SIZE + local_id.y;
    uint col = group_id.x * TILE_SIZE + local_id.x;
    
    float sum = 0.0f;
    
    // Use threadgroup memory for shared tiles
    threadgroup float asub[TILE_SIZE * TILE_SIZE];
    threadgroup float bsub[TILE_SIZE * TILE_SIZE];
    
    for (uint t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load A tile
        uint a_row = row;
        uint a_col = t * TILE_SIZE + local_id.x;
        if (a_row < m && a_col < k) {
            asub[local_id.y * TILE_SIZE + local_id.x] = a[a_row * k + a_col];
        } else {
            asub[local_id.y * TILE_SIZE + local_id.x] = 0.0f;
        }
        
        // Load B tile  
        uint b_row = t * TILE_SIZE + local_id.y;
        uint b_col = col;
        if (b_row < k && b_col < n) {
            bsub[local_id.y * TILE_SIZE + local_id.x] = b[b_row * n + b_col];
        } else {
            bsub[local_id.y * TILE_SIZE + local_id.x] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial sum
        for (uint p = 0; p < TILE_SIZE; p++) {
            sum += asub[local_id.y * TILE_SIZE + p] * bsub[p * TILE_SIZE + local_id.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

kernel void softmax_forward(
    device const float* logits     [[buffer(0)]],
    device float*       output     [[buffer(1)]],
    constant uint&      batch_size [[buffer(2)]],
    constant uint&      num_classes [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < batch_size) {
        uint offset = gid * num_classes;
        
        // Find max
        float max_logit = logits[offset];
        for (uint i = 1; i < num_classes; i++) {
            max_logit = max(max_logit, logits[offset + i]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (uint i = 0; i < num_classes; i++) {
            float exp_val = exp(logits[offset + i] - max_logit);
            output[offset + i] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (uint i = 0; i < num_classes; i++) {
            output[offset + i] /= sum_exp;
        }
    }
}

kernel void softmax_xent(
    device const float* logits       [[buffer(0)]],
    device const float* labels       [[buffer(1)]],
    device float*       loss         [[buffer(2)]],
    device float*       grad_logits  [[buffer(3)]],
    constant uint&      batch_size   [[buffer(4)]],
    constant uint&      num_classes  [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < batch_size) {
        uint offset = gid * num_classes;
        uint label = uint(labels[gid]);
        
        // Find max for numerical stability
        float max_logit = logits[offset];
        for (uint i = 1; i < num_classes; i++) {
            max_logit = max(max_logit, logits[offset + i]);
        }
        
        // Compute softmax probabilities
        float sum_exp = 0.0f;
        for (uint i = 0; i < num_classes; i++) {
            float exp_val = exp(logits[offset + i] - max_logit);
            grad_logits[offset + i] = exp_val;
            sum_exp += exp_val;
        }
        
        // Compute loss for this sample
        float prob_true_class = grad_logits[offset + label] / sum_exp;
        float sample_loss = -log(prob_true_class + 1e-7f);
        
        // Store per-sample loss (will be reduced later on CPU)
        loss[gid] = sample_loss;
        
        // Compute gradients: prob - target (scaled by 1/batch_size)
        for (uint i = 0; i < num_classes; i++) {
            float prob = grad_logits[offset + i] / sum_exp;
            grad_logits[offset + i] = (prob - (i == label ? 1.0f : 0.0f)) / float(batch_size);
        }
    }
}

// Gradient computation for weights: dW = X^T * dY
kernel void matmul_grad_w(
    device const float* x         [[buffer(0)]], // Input (batch_size, in_features)
    device const float* dy        [[buffer(1)]], // Output gradients (batch_size, out_features)  
    device float*       dw        [[buffer(2)]], // Weight gradients (in_features, out_features)
    constant uint3&     dims      [[buffer(3)]], // batch_size, in_features, out_features
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_size = dims.x;
    uint in_features = dims.y; 
    uint out_features = dims.z;
    
    uint i = gid.y; // in_features
    uint j = gid.x; // out_features
    
    if (i < in_features && j < out_features) {
        float sum = 0.0f;
        for (uint b = 0; b < batch_size; b++) {
            sum += x[b * in_features + i] * dy[b * out_features + j];
        }
        dw[i * out_features + j] += sum; // Accumulate gradients
    }
}

// Gradient computation for bias: db = sum(dY, axis=0)
kernel void bias_grad(
    device const float* dy        [[buffer(0)]], // Output gradients (batch_size, out_features)
    device float*       db        [[buffer(1)]], // Bias gradients (out_features,)
    constant uint2&     dims      [[buffer(2)]], // batch_size, out_features
    uint gid [[thread_position_in_grid]]
) {
    uint batch_size = dims.x;
    uint out_features = dims.y;
    
    if (gid < out_features) {
        float sum = 0.0f;
        for (uint b = 0; b < batch_size; b++) {
            sum += dy[b * out_features + gid];
        }
        db[gid] += sum; // Accumulate gradients
    }
}

// Gradient computation for input: dX = dY * W^T
kernel void matmul_grad_x(
    device const float* dy        [[buffer(0)]], // Output gradients (batch_size, out_features)
    device const float* w         [[buffer(1)]], // Weights (in_features, out_features)
    device float*       dx        [[buffer(2)]], // Input gradients (batch_size, in_features)
    constant uint3&     dims      [[buffer(3)]], // batch_size, out_features, in_features
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_size = dims.x;
    uint out_features = dims.y;
    uint in_features = dims.z;
    
    uint b = gid.y; // batch_size
    uint i = gid.x; // in_features
    
    if (b < batch_size && i < in_features) {
        float sum = 0.0f;
        for (uint j = 0; j < out_features; j++) {
            sum += dy[b * out_features + j] * w[i * out_features + j];
        }
        dx[b * in_features + i] = sum;
    }
}

// im2col kernel: convert image patches to columns for convolution as matrix multiplication
// Input: (N, C, H, W) -> Output: (kernel_h * kernel_w * input_channels, output_h * output_w * N)
kernel void im2col(
    device const float* input       [[buffer(0)]], // Input tensor (N, C, H, W)
    device float*       output      [[buffer(1)]], // Output columns
    constant uint4&     input_dims  [[buffer(2)]], // N, C, H, W
    constant uint2&     kernel_size [[buffer(3)]], // kernel_h, kernel_w  
    constant uint2&     padding     [[buffer(4)]], // pad_h, pad_w
    constant uint2&     stride      [[buffer(5)]], // stride_h, stride_w
    constant uint2&     output_dims [[buffer(6)]], // output_h, output_w
    uint2 gid [[thread_position_in_grid]]
) {
    uint N = input_dims.x;
    uint C = input_dims.y; 
    uint H = input_dims.z;
    uint W = input_dims.w;
    
    uint kernel_h = kernel_size.x;
    uint kernel_w = kernel_size.y;
    uint pad_h = padding.x;
    uint pad_w = padding.y;
    uint stride_h = stride.x;
    uint stride_w = stride.y;
    uint output_h = output_dims.x;
    uint output_w = output_dims.y;
    
    uint col_idx = gid.x; // column index in output matrix
    uint row_idx = gid.y; // row index in output matrix
    
    uint total_cols = output_h * output_w * N;
    uint total_rows = kernel_h * kernel_w * C;
    
    if (col_idx >= total_cols || row_idx >= total_rows) return;
    
    // Decode column index: which spatial position and batch
    uint spatial_idx = col_idx % (output_h * output_w);
    uint batch_idx = col_idx / (output_h * output_w);
    uint out_y = spatial_idx / output_w;
    uint out_x = spatial_idx % output_w;
    
    // Decode row index: which kernel position and channel
    uint kernel_spatial = row_idx % (kernel_h * kernel_w);
    uint channel_idx = row_idx / (kernel_h * kernel_w);
    uint kernel_y = kernel_spatial / kernel_w;
    uint kernel_x = kernel_spatial % kernel_w;
    
    // Compute input position
    int in_y = int(out_y * stride_h) - int(pad_h) + int(kernel_y);
    int in_x = int(out_x * stride_w) - int(pad_w) + int(kernel_x);
    
    // Check bounds and set value
    if (in_y >= 0 && in_y < int(H) && in_x >= 0 && in_x < int(W)) {
        uint input_idx = batch_idx * (C * H * W) + channel_idx * (H * W) + uint(in_y) * W + uint(in_x);
        output[row_idx * total_cols + col_idx] = input[input_idx];
    } else {
        output[row_idx * total_cols + col_idx] = 0.0f;
    }
}

// MaxPool2d forward pass  
kernel void maxpool2d_forward(
    device const float* input       [[buffer(0)]], // Input (N, C, H, W)
    device float*       output      [[buffer(1)]], // Output (N, C, H_out, W_out)
    device uint*        indices     [[buffer(2)]], // Argmax indices for backward
    constant uint4&     input_dims  [[buffer(3)]], // N, C, H, W
    constant uint2&     kernel_size [[buffer(4)]], // kernel_h, kernel_w
    constant uint2&     stride      [[buffer(5)]], // stride_h, stride_w
    constant uint2&     output_dims [[buffer(6)]], // output_h, output_w
    uint3 gid [[thread_position_in_grid]]
) {
    uint N = input_dims.x;
    uint C = input_dims.y;
    uint H = input_dims.z;
    uint W = input_dims.w;
    
    uint kernel_h = kernel_size.x;
    uint kernel_w = kernel_size.y;
    uint stride_h = stride.x;
    uint stride_w = stride.y;
    uint output_h = output_dims.x;
    uint output_w = output_dims.y;
    
    uint n = gid.z;
    uint c = gid.y; 
    uint out_idx = gid.x;
    
    if (n >= N || c >= C || out_idx >= (output_h * output_w)) return;
    
    uint out_y = out_idx / output_w;
    uint out_x = out_idx % output_w;
    
    uint start_y = out_y * stride_h;
    uint start_x = out_x * stride_w;
    
    float max_val = -INFINITY;
    uint max_idx = 0;
    
    for (uint ky = 0; ky < kernel_h; ky++) {
        for (uint kx = 0; kx < kernel_w; kx++) {
            uint in_y = start_y + ky;
            uint in_x = start_x + kx;
            
            if (in_y < H && in_x < W) {
                uint input_idx = n * (C * H * W) + c * (H * W) + in_y * W + in_x;
                if (input[input_idx] > max_val) {
                    max_val = input[input_idx];
                    max_idx = input_idx;
                }
            }
        }
    }
    
    uint output_idx = n * (C * output_h * output_w) + c * (output_h * output_w) + out_y * output_w + out_x;
    output[output_idx] = max_val;
    indices[output_idx] = max_idx;
}

kernel void sgd_step(
    device float*       params    [[buffer(0)]],
    device const float* grads     [[buffer(1)]],
    constant float&     lr        [[buffer(2)]],
    constant uint&      n         [[buffer(3)]],
    device float*       velocity  [[buffer(4)]],
    constant float&     momentum  [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        if (velocity != nullptr) {
            velocity[gid] = momentum * velocity[gid] + lr * grads[gid];
            params[gid] -= velocity[gid];
        } else {
            params[gid] -= lr * grads[gid];
        }
    }
}

// M6 Advanced Optimization: Fused ReLU + Linear forward pass
kernel void linear_relu_forward(
    device const float* x    [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device const float* b    [[buffer(2)]],
    device float* y          [[buffer(3)]],
    constant uint3& dims     [[buffer(4)]], // batch, in_features, out_features
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.y;
    uint out_idx = gid.x;
    uint batch = dims.x;
    uint in_features = dims.y;
    uint out_features = dims.z;
    
    if (batch_idx < batch && out_idx < out_features) {
        float sum = b[out_idx];
        for (uint i = 0; i < in_features; i++) {
            sum += x[batch_idx * in_features + i] * w[i * out_features + out_idx];
        }
        y[batch_idx * out_features + out_idx] = max(0.0f, sum); // Fused ReLU
    }
}

// M6 Advanced Optimization: Mixed precision GEMM (FP16 computation with FP32 accumulation)
kernel void gemm_fp16_fp32(
    device const half* a      [[buffer(0)]],
    device const half* b      [[buffer(1)]],
    device float* c           [[buffer(2)]],
    constant uint3& dims      [[buffer(3)]], // m, n, k
    constant float& alpha     [[buffer(4)]],
    constant float& beta      [[buffer(5)]],
    device const float* bias  [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = dims.x;
    uint n = dims.y;
    uint k = dims.z;
    
    uint i = gid.y;
    uint j = gid.x;
    
    if (i < m && j < n) {
        float sum = 0.0f; // Use FP32 for accumulation
        for (uint p = 0; p < k; p++) {
            // Load FP16 values and convert to FP32 for computation
            float a_val = float(a[i * k + p]);
            float b_val = float(b[p * n + j]);
            sum += a_val * b_val;
        }
        
        c[i * n + j] = alpha * sum + beta * c[i * n + j];
        
        if (bias != nullptr) {
            c[i * n + j] += bias[j];
        }
    }
}

// M6 Advanced Optimization: Fused Convolution + BatchNorm + ReLU (placeholder structure)
kernel void conv_bn_relu_fused(
    device const float* input     [[buffer(0)]],
    device const float* weights   [[buffer(1)]],
    device const float* bn_weight [[buffer(2)]],
    device const float* bn_bias   [[buffer(3)]],
    device const float* bn_mean   [[buffer(4)]],
    device const float* bn_var    [[buffer(5)]],
    device float* output          [[buffer(6)]],
    constant uint4& input_dims    [[buffer(7)]], // N, C, H, W
    constant uint4& output_dims   [[buffer(8)]], // N, C_out, H_out, W_out
    constant float& eps           [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // This would implement fused convolution + batch normalization + ReLU
    // For now, it's a placeholder showing the kernel signature
    // A full implementation would require im2col + GEMM + BN + ReLU in one kernel
    
    uint n = gid.z;
    uint c = gid.y;
    uint spatial_idx = gid.x;
    
    if (n < output_dims.x && c < output_dims.y && spatial_idx < (output_dims.z * output_dims.w)) {
        // Placeholder - would implement the full fused operation
        output[n * (output_dims.y * output_dims.z * output_dims.w) + 
               c * (output_dims.z * output_dims.w) + spatial_idx] = 0.0f;
    }
}

// M6 Advanced Optimization: Optimized reduction for loss computation
kernel void reduce_sum_loss(
    device const float* losses    [[buffer(0)]], // Per-sample losses
    device float* total_loss      [[buffer(1)]], // Output total loss
    constant uint& batch_size     [[buffer(2)]],
    threadgroup float* shared     [[threadgroup(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Load losses into shared memory
    float local_sum = 0.0f;
    for (uint i = lid; i < batch_size; i += group_size) {
        local_sum += losses[i];
    }
    shared[lid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (lid == 0) {
        total_loss[0] = shared[0] / float(batch_size);
    }
}