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

template <Activation type>
__device__ __forceinline__ float activate_fwd(float x) {
    if constexpr (type == Activation::ReLU) {
        return fmaxf(0.0f, x);
    } else if constexpr (type == Activation::ClampedReLU) {
        return clamp(x, 0.0f, 1.0f);
    } else if constexpr (type == Activation::SquaredClampedReLU) {
        x = clamp(x, 0.0f, 1.0f);
        return x * x;
    } else if constexpr (type == Activation::Sigmoid) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        return x;
    }
}

// assumes x is activated
template <Activation type>
__device__ __forceinline__ float activate_bwd(float x) {
    if constexpr (type == Activation::ReLU) {
        return (x > 0.0f) ? 1.0f : 0.0f;
    } else if constexpr (type == Activation::ClampedReLU) {
        return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f;
    } else if constexpr (type == Activation::SquaredClampedReLU) {
        return (x > 0.0f && x < 1.0f) ? 2.0f * x : 0.0f;
    } else if constexpr (type == Activation::Sigmoid) {
        return x * (1.0f - x);
    } else {
        return 1.0f;
    }
}

// Helper macro to dispatch activation types to templated kernel calls
// Usage: DISPATCH_ACTIVATION(act_type, MY_KERNEL_CALL, kernel_args...)
// Expands to: MY_KERNEL_CALL<Activation::Type> kernel_args
#define DISPATCH_ACTIVATION(act_type, KERNEL_NAME, ...)                                                                \
    switch (act_type) {                                                                                                \
    case Activation::Linear:                                                                                           \
        KERNEL_NAME<Activation::Linear> __VA_ARGS__;                                                                   \
        break;                                                                                                         \
    case Activation::ReLU:                                                                                             \
        KERNEL_NAME<Activation::ReLU> __VA_ARGS__;                                                                     \
        break;                                                                                                         \
    case Activation::ClampedReLU:                                                                                      \
        KERNEL_NAME<Activation::ClampedReLU> __VA_ARGS__;                                                              \
        break;                                                                                                         \
    case Activation::SquaredClampedReLU:                                                                               \
        KERNEL_NAME<Activation::SquaredClampedReLU> __VA_ARGS__;                                                       \
        break;                                                                                                         \
    case Activation::Sigmoid:                                                                                          \
        KERNEL_NAME<Activation::Sigmoid> __VA_ARGS__;                                                                  \
        break;                                                                                                         \
    }

} // namespace kernel
