#pragma once

#include "../../data/include.h"
#include "../util.h"

enum class Activation {
    Linear,
    ReLU,
    ClampedReLU,
    SquaredClampedReLU,
    Sigmoid,
};

namespace kernel {

__device__ __forceinline__ float activate_fwd(float x, const Activation type) {
    switch (type) {
    case Activation::ReLU:
        return fmaxf(0.0f, x);
    case Activation::ClampedReLU:
        return clamp(x, 0.0f, 1.0f);
    case Activation::SquaredClampedReLU:
        x = clamp(x, 0.0f, 1.0f);
        return x * x;
    case Activation::Sigmoid:
        return 1.0f / (1.0f + expf(-x));
    default:
        return x;
    }
}

// assumes x is activated
__device__ __forceinline__ float activate_bwd(float x, const Activation type) {
    switch (type) {
    case Activation::ReLU:
        return (x > 0.0f) ? 1.0f : 0.0f;
    case Activation::ClampedReLU:
        return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f;
    case Activation::SquaredClampedReLU:
        return (x > 0.0f && x < 1.0f) ? 2.0f * x : 0.0f;
    case Activation::Sigmoid:
        return x * (1.0f - x);
    default:
        return 1.0f;
    }
}

} // namespace kernel
