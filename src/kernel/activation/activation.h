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

inline bool has_activation(const ActivationType type) {
    return type != ActivationType::Linear;
}

namespace kernel {

inline __device__ float activate_fwd(float x, const ActivationType type) {
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

inline __device__ float activate_bwd(float x, const ActivationType type) {
    switch(type) {
    case ActivationType::ReLU:
        return (x > 0.0f) ? 1.0f : 0.0f;
    case ActivationType::CReLU:
        return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f;
    case ActivationType::SCReLU:
        return (x > 0.0f && x < 1.0f) ? 2.0f * x : 0.0f;
    case ActivationType::Sigmoid:
        x = activate_fwd(x, ActivationType::Sigmoid);
        return x * (1 - x);
    default:
        return 1.0f; // None
    }
}

} // namespace kernel
