#include "loss.h"

__global__ void mpe_kernel( //
    const float *targets,
    const float *output_v,
    float *output_g,
    float *loss,
    const float power,
    const ActivationType act_type,
    const int size //
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    float pre_activated = output_v[idx];
    float activated = activate(pre_activated, act_type);
    float diff = activated - targets[idx];
    float abs_diff = abs(diff);

    float grad = powf(abs_diff, power - 1.0f) * power * activationDer(pre_activated, act_type);
    output_g[idx] = (diff > 0 ? 1 : -1) * grad;

    atomicAdd(loss, powf(abs_diff, power));
}

__global__ void mse_kernel( //
    const float *targets,
    const float *output_v,
    float *output_g,
    float *loss,
    const ActivationType act_type,
    const int size //
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    float pre_activated = output_v[idx];
    float activated = activate(output_v[idx], act_type);
    float diff = activated - targets[idx];
    output_g[idx] = 2.0f * diff * activationDer(pre_activated, act_type);

    atomicAdd(loss, diff * diff);
}
