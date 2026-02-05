#pragma once

#include "../../data/include.h"
#include "../util.h"

enum class Activation {
    Linear,
    ReLU,
    CReLU,
    SCReLU,
    Sigmoid,
};

inline __host__ __device__ bool has_activation(const Activation type) {
    return type != Activation::Linear;
}

namespace kernel {

inline __device__ float activate_fwd(float x, const Activation type) {
    switch (type) {
    case Activation::ReLU:
        return fmaxf(0.0f, x);
    case Activation::CReLU:
        return clamp(x, 0.0f, 1.0f);
    case Activation::SCReLU:
        x = clamp(x, 0.0f, 1.0f);
        return x * x;
    case Activation::Sigmoid:
        return 1.0f / (1.0f + expf(-x));
    default:
        return x; // Linear
    }
}

inline __device__ float activate_bwd(float x, const Activation type) {
    switch (type) {
    case Activation::ReLU:
        return (x > 0.0f) ? 1.0f : 0.0f;
    case Activation::CReLU:
        return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f;
    case Activation::SCReLU:
        return (x > 0.0f && x < 1.0f) ? 2.0f * x : 0.0f;
    case Activation::Sigmoid:
        x = activate_fwd(x, Activation::Sigmoid);
        return x * (1 - x);
    default:
        return 1.0f; // Linear
    }
}

} // namespace kernel
