#pragma once

#include "../../nn/data/include.h"

struct OptimParams {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float decay;

    OptimParams(float lr, float beta1, float beta2, float eps, float decay)
        : lr(lr), beta1(beta1), beta2(beta2), eps(eps), decay(decay) {}
};

void adam_optim( //
    DenseMatrix<float> &vals,
    DenseMatrix<float> &grads,
    Array<float> &moms,
    Array<float> &vels,
    const OptimParams &params,
    const float min_val,
    const float max_val,
    const float grad_scale);
