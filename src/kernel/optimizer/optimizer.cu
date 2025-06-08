#include "optimizer.h"
#include <iostream>

__device__ float clamp(float x, float min, float max) {
    return fmaxf(min, fminf(x, max));
}

// clang-format off
__global__ void adam_kernel
(
    float *vals, 
    float *grads, 
    float *moms, 
    float *vels, 
    const float lr, 
    const float beta1,
    const float beta2, 
    const float eps, 
    const float decay, 
    const float min_val,
    const float max_val, 
    const float grad_scale, 
    const int size
) {
    // clang-format on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    const float grad = grads[idx] * grad_scale;
    float mom = moms[idx];
    float vel = vels[idx];
    float val = vals[idx];

    mom = beta1 * mom + (1.0f - beta1) * grad;
    vel = beta2 * vel + (1.0f - beta2) * grad * grad;

    val *= decay;
    val -= lr * mom / (sqrtf(vel) + eps);

    moms[idx] = mom;
    vels[idx] = vel;
    vals[idx] = clamp(val, min_val, max_val);
    grads[idx] = 0.0f;
}

// https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py
// clang-format off
__global__ void radam_kernel
(
    float *vals, 
    float *grads, 
    float *moms, 
    float *vels, 
    const float lr, 
    const float beta1,
    const float beta2, 
    const float eps, 
    const float decay, 
    const float min_val,
    const float max_val, 
    const float grad_scale, 
    const int N_sma_threshold, 
    const int step,
    const int size
) {
    // clang-format on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    const float grad = grads[idx] * grad_scale;
    float mom = moms[idx];
    float vel = vels[idx];
    float val = vals[idx];

    mom = beta1 * mom + (1.0f - beta1) * grad;
    vel = beta2 * vel + (1.0f - beta2) * grad * grad;

    val *= decay;

    float beta2_t = powf(beta2, step);
    float N_sma_max = 2.0f / (1.0f - beta2) - 1.0f;
    float N_sma = N_sma_max - 2.0f * step * beta2_t / (1.0f - beta2_t);

    if(N_sma >= N_sma_threshold) {
        // clang-format off
        float step_size = lr 
                        * sqrtf((1.0f - beta2_t) * (N_sma - 4.0f) / (N_sma_max - 4.0f) 
                        * (N_sma - 2.0f) / N_sma * N_sma_max / (N_sma_max - 2.0f)) 
                        / (1.0f - powf(beta1, step));
        // clang-format on
        float denom = sqrtf(vel) + eps;
        val -= step_size * mom / denom;
    } else {
        float step_size = lr * (1.0f - powf(beta1, step));
        val -= step_size * mom;
    }

    moms[idx] = mom;
    vels[idx] = vel;
    vals[idx] = clamp(val, min_val, max_val);
    grads[idx] = 0.0f;
}

// https://github.com/official-stockfish/nnue-pytorch/blob/master/ranger.py
// clang-format off
__global__ void ranger_kernel
(
    float *vals, 
    float *grads, 
    float *moms, 
    float *vels, 
    float *slow_buffer, 
    const float lr,
    const float beta1, 
    const float beta2, 
    const float eps, 
    const float decay,
    const float min_val, 
    const float max_val, 
    const float grad_scale, 
    const float alpha,
    const int k, 
    const int N_sma_threshold, 
    const int step, 
    const int size
) {
    // clang-format on
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    const float grad = grads[idx] * grad_scale;
    float mom = moms[idx];
    float vel = vels[idx];
    float val = vals[idx];

    mom = beta1 * mom + (1.0f - beta1) * grad;
    vel = beta2 * vel + (1.0f - beta2) * grad * grad;

    val *= decay;

    float beta2_t = powf(beta2, step);
    float beta1_correction = 1.0f - powf(beta1, step);
    float N_sma_max = 2.0f / (1.0f - beta2) - 1.0f;
    float N_sma = N_sma_max - 2.0f * step * beta2_t / (1.0f - beta2_t);

    if(N_sma >= N_sma_threshold) {
        // clang-format off
        float step_size = lr 
                        * sqrtf((1.0f - beta2_t) * (N_sma - 4.0f) / (N_sma_max - 4.0f) 
                        * (N_sma - 2.0f) / N_sma * N_sma_max / (N_sma_max - 2.0f)) 
                        / beta1_correction;
        // clang-format on
        val -= step_size * mom / (sqrtf(vel) + eps);
    } else
        val -= (lr / beta1_correction) * mom;

    // moving average of weights
    if(step % k == 0) {
        float slow = slow_buffer[idx];
        slow += alpha * (val - slow);
        slow_buffer[idx] = slow;
        val = slow;
    }

    moms[idx] = mom;
    vels[idx] = vel;
    vals[idx] = clamp(val, min_val, max_val);
    grads[idx] = 0.0f;
}
