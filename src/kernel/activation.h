#pragma once

#include "../nn/data.h"

enum ActivationType { Linear, ReLU, CReLU, SCReLU, Sigmoid };

// helper function to get the activation function

inline __device__ float activate(float x, ActivationType type) {
    switch(type) {
    case ReLU:
        return max(0.0f, x);
    case CReLU:
        return min(1.0f, max(0.0f, x));
    case SCReLU:
        x = min(1.0f, max(0.0f, x));
        return x * x;
    case Sigmoid:
        return 1.0f / (1.0f + expf(-x));
    default:
        return x; // Linear
    }
}

inline __device__ float activationDer(float x, ActivationType type) {
    switch(type) {
    case ReLU:
        return x > 0.0f ? 1.0f : 0.0f;
    case CReLU:
        return x > 0.0f && x < 1.0f ? 1.0f : 0.0f;
    case SCReLU:
        return x > 0.0f && x < 1.0f ? 2.0f * x : 0.0f;
    case Sigmoid:
        return x * (1 - x); // assumes x is already sigmoided
    default:
        return 1.0f; // Linear
    }
}
