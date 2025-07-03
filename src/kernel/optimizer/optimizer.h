#pragma once

#include "../../nn/data.h"

struct AdamParams {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float decay;

    AdamParams(float lr, float beta1, float beta2, float eps, float decay)
        : lr(lr), beta1(beta1), beta2(beta2), eps(eps), decay(decay) {}
};

void adam_optim( //
    DenseMatrix &vals,
    DenseMatrix &grads,
    Array<float> &moms,
    Array<float> &vels,
    const AdamParams &params,
    const float min_val,
    const float max_val,
    const float grad_scale);

void radam_optim( //
    DenseMatrix &vals,
    DenseMatrix &grads,
    Array<float> &moms,
    Array<float> &vels,
    const AdamParams &params,
    const float min_val,
    const float max_val,
    const float grad_scale,
    const int N_sma_threshold,
    const int step);

void ranger_optim( //
    DenseMatrix &vals,
    DenseMatrix &grads,
    Array<float> &moms,
    Array<float> &vels,
    Array<float> &slow_buffer,
    const AdamParams &params,
    const float min_val,
    const float max_val,
    const float grad_scale,
    const float alpha,
    const int k,
    const int N_sma_threshold,
    const int step);
