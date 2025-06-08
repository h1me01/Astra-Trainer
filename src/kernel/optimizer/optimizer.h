#pragma once

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
);

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
);

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
);
// clang-format on