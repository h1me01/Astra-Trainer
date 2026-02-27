#pragma once

#include "../../data/include.h"
#include "../util.h"

enum class ActivationType {
    Linear,
    ReLU,
    ClippedReLU,
    SqrClippedReLU,
    Sigmoid,
};

namespace kernel {

#define DISPATCH_ACTIVATION(act_type, KERNEL_NAME, ...)                                                                \
    switch (act_type) {                                                                                                \
    case ActivationType::Linear:                                                                                       \
        KERNEL_NAME<ActivationType::Linear> __VA_ARGS__;                                                               \
        break;                                                                                                         \
    case ActivationType::ReLU:                                                                                         \
        KERNEL_NAME<ActivationType::ReLU> __VA_ARGS__;                                                                 \
        break;                                                                                                         \
    case ActivationType::ClippedReLU:                                                                                  \
        KERNEL_NAME<ActivationType::ClippedReLU> __VA_ARGS__;                                                          \
        break;                                                                                                         \
    case ActivationType::SqrClippedReLU:                                                                               \
        KERNEL_NAME<ActivationType::SqrClippedReLU> __VA_ARGS__;                                                       \
        break;                                                                                                         \
    case ActivationType::Sigmoid:                                                                                      \
        KERNEL_NAME<ActivationType::Sigmoid> __VA_ARGS__;                                                              \
        break;                                                                                                         \
    }

template <ActivationType type>
__device__ __forceinline__ float activate_fwd(float x) {
    if constexpr (type == ActivationType::ReLU) {
        return fmaxf(0.0f, x);
    } else if constexpr (type == ActivationType::ClippedReLU) {
        return clamp(x, 0.0f, 1.0f);
    } else if constexpr (type == ActivationType::SqrClippedReLU) {
        x = clamp(x, 0.0f, 1.0f);
        return x * x;
    } else if constexpr (type == ActivationType::Sigmoid) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        return x;
    }
}

template <ActivationType type, bool fused = false>
__device__ __forceinline__ float activate_bwd(float x) {
    if constexpr (type == ActivationType::ReLU) {
        return (x > 0.0f) ? 1.0f : 0.0f;
    } else if constexpr (type == ActivationType::ClippedReLU) {
        return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f;
    } else if constexpr (type == ActivationType::SqrClippedReLU) {
        if (x > 0.0f && x < 1.0f)
            return 2.0f * (fused ? sqrtf(x) : x);
        else
            return 0.0f;
    } else if constexpr (type == ActivationType::Sigmoid) {
        if (!fused)
            x = activate_fwd<ActivationType::Sigmoid>(x);
        return x * (1.0f - x);
    } else {
        return 1.0f;
    }
}

template <ActivationType act_type>
__device__ __forceinline__ float4& activate_fwd_f4(float4& a) {
    a.x = activate_fwd<act_type>(a.x);
    a.y = activate_fwd<act_type>(a.y);
    a.z = activate_fwd<act_type>(a.z);
    a.w = activate_fwd<act_type>(a.w);
    return a;
}

void activation_fwd(const DenseMatrix& in_d, DenseMatrix& out_d, const ActivationType type);
void activation_bwd(Tensor& in, const DenseMatrix& out_g, const ActivationType type);

} // namespace kernel
