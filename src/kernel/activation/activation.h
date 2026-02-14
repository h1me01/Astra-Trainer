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

template <Activation type, bool fused = false>
__device__ __forceinline__ float activate_bwd(float x) {
    if constexpr (type == Activation::ReLU) {
        return (x > 0.0f) ? 1.0f : 0.0f;
    } else if constexpr (type == Activation::ClampedReLU) {
        return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f;
    } else if constexpr (type == Activation::SquaredClampedReLU) {
        if (x > 0.0f && x < 1.0f)
            return 2.0f * (fused ? sqrtf(x) : x);
        else
            return 0.0f;
    } else if constexpr (type == Activation::Sigmoid) {
        if (!fused)
            x = activate_fwd<Activation::Sigmoid>(x);
        return x * (1.0f - x);
    } else {
        return 1.0f;
    }
}

void activation_fwd(const DenseMatrix& in_v, DenseMatrix& out_v, const Activation type);
void activation_bwd(Tensor& in, const DenseMatrix& out_g, const Activation type);

} // namespace kernel
