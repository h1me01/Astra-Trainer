#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

namespace optim_utils {

struct RAdamParams {
    float beta1_t;
    float beta2_t;
    float N_sma;
    float N_sma_max;
};

inline float get_decay(const float lr, const float decay) {
    return 1.0f - lr * decay;
}

inline __device__ float get_radam_update(float mom, float vel, float lr, float eps, RAdamParams params) {
    if (params.N_sma >= 5) {
        const float step_size = lr *
                                sqrtf(
                                    (1.0f - params.beta2_t) * (params.N_sma - 4) / (params.N_sma_max - 4) *
                                    (params.N_sma - 2) / params.N_sma * params.N_sma_max / (params.N_sma_max - 2)
                                ) /
                                (1.0f - params.beta1_t);
        const float denom = sqrtf(vel) + eps;
        return step_size * (mom / denom);
    } else {
        const float step_size = lr / (1.0f - params.beta1_t);
        return step_size * mom;
    }
}

inline RAdamParams get_radam_params(const float beta1, const float beta2, const int step) {
    RAdamParams params;
    params.beta1_t = std::pow(beta1, step);
    params.beta2_t = std::pow(beta2, step);
    params.N_sma_max = 2 / (1 - beta2) - 1;
    params.N_sma = params.N_sma_max - 2 * step * params.beta2_t / (1 - params.beta2_t);
    return params;
}

} // namespace optim_utils

void adam_optim(
    Tensor& param,
    Array<float>& moms,
    Array<float>& vels,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float decay,
    const float grad_scale
);

void radam_optim(
    Tensor& param,
    Array<float>& moms,
    Array<float>& vels,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float decay,
    const float grad_scale,
    const optim_utils::RAdamParams radam_params,
    const int step
);

void ranger_optim(
    Tensor& param,
    Array<float>& moms,
    Array<float>& vels,
    Array<float>& slow_buffer,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float decay,
    const float grad_scale,
    const optim_utils::RAdamParams radam_params,
    const int step
);

} // namespace kernel
