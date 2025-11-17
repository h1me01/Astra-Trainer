#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

namespace optimizer_utils {

inline float get_decay(const float lr, const float decay) {
    return 1.0f - lr * decay;
}

inline __device__ float get_radam_update( //
    float mom,                            //
    float vel,                            //
    float lr,                             //
    float eps,                            //
    float beta1_t,                        //
    float beta2_t,                        //
    int N_sma,                            //
    int N_sma_max                         //
) {
    if(N_sma >= 5) {
        const float step_size =
            lr *
            sqrtf((1.0f - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) /
            (1.0f - beta1_t);
        const float denom = sqrtf(vel) + eps;
        return step_size * (mom / denom);
    } else {
        const float step_size = lr / (1.0f - beta1_t);
        return step_size * mom;
    }
}

} // namespace optimizer_utils

void adam_optim( //
    Tensor &param,
    Array<float> &moms,
    Array<float> &vels,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float decay,
    const float grad_scale);

void radam_optim( //
    Tensor &param,
    Array<float> &moms,
    Array<float> &vels,
    const float lr,
    const float beta1,
    const float beta2,
    const float beta1_t,
    const float beta2_t,
    const float eps,
    const float decay,
    const float grad_scale,
    const int N_sma,
    const int N_sma_max,
    const int step);

void ranger_optim( //
    Tensor &param,
    Array<float> &moms,
    Array<float> &vels,
    Array<float> &slow_buffer,
    const float lr,
    const float beta1,
    const float beta2,
    const float beta1_t,
    const float beta2_t,
    const float eps,
    const float decay,
    const float grad_scale,
    const int N_sma,
    const int N_sma_max,
    const int step);

} // namespace kernel
