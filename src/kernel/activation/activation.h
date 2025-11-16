#pragma once

#include "../../data/include.h"
#include "../util.h"

enum class ActivationType {
    Linear,
    ReLU,
    CReLU,
    SCReLU,
    Sigmoid,
};

namespace kernel {

inline __device__ float activate(float x, const ActivationType type) {
    switch(type) {
    case ActivationType::ReLU:
        return fmaxf(0.0f, x);
    case ActivationType::CReLU:
        return clamp(x, 0.0f, 1.0f);
    case ActivationType::SCReLU:
        x = clamp(x, 0.0f, 1.0f);
        return x * x;
    case ActivationType::Sigmoid:
        return 1.0f / (1.0f + expf(-x));
    default:
        return x; // None
    }
}

inline __device__ float activate_der(float x, const ActivationType type) {
    switch(type) {
    case ActivationType::ReLU:
        return (x > 0.0f) ? 1.0f : 0.0f;
    case ActivationType::CReLU:
        return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f;
    case ActivationType::SCReLU:
        return (x > 0.0f && x < 1.0f) ? 2.0f * x : 0.0f;
    case ActivationType::Sigmoid:
        x = activate(x, ActivationType::Sigmoid);
        return x * (1 - x);
    default:
        return 1.0f; // None
    }
}

void activate_fwd( //
    const DenseMatrix<float> &in_v,
    DenseMatrix<float> &out_v,
    const ActivationType type);

void activate_bwd( //
    Tensor<float> &in,
    const DenseMatrix<float> &out_g,
    const ActivationType type);

} // namespace kernel
