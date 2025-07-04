#pragma once

#include "../nn/data/include.h"
#include "util.h"

enum ActivationType { //
    Linear,
    ReLU,
    CReLU,
    SReLU,
    SCReLU,
    Sigmoid,
    Tanh
};

inline __device__ float activate(float x, ActivationType type) {
    switch(type) {
    case ReLU:
        return max(0.0f, x);
    case CReLU:
        return clamp(x, 0.0f, 1.0f);
    case SReLU:
        return (x > 0.0f) ? x * x : 0.0f;
    case SCReLU:
        x = clamp(x, 0.0f, 1.0f);
        return x * x;
    case Sigmoid:
        return 1.0f / (1.0f + expf(-x));
    case Tanh:
        return tanhf(x);
    default:
        return x; // Linear
    }
}

inline __device__ float activate_der(float x, ActivationType type) {
    switch(type) {
    case ReLU:
        return (x > 0.0f) ? 1.0f : 0.0f;
    case CReLU:
        return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f;
    case SReLU:
        return (x > 0.0f) ? 2.0f * x : 0.0f;
    case SCReLU:
        return (x > 0.0f && x < 1.0f) ? 2.0f * x : 0.0f;
    case Sigmoid:
        x = activate(x, Sigmoid);
        return x * (1 - x);
    case Tanh:
        x = activate(x, Tanh);
        return 1.0f - x * x;
    default:
        return 1.0f; // Linear
    }
}
