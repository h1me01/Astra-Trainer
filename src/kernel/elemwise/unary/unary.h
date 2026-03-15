#pragma once

#include <variant>

#include "../../../data/include.h"
#include "../../util.h"

namespace kernel {

struct Linear {
    __device__ float forward(float x) const { return x; }
    template <bool fused = false>
    __device__ float backward(float x) const {
        return 1.0f;
    }
};

struct ReLU {
    __device__ float forward(float x) const { return max(0.0f, x); }
    template <bool fused = false>
    __device__ float backward(float x) const {
        return (x > 0.0f) ? 1.0f : 0.0f;
    }
};

struct ClippedReLU {
    __device__ float forward(float x) const { return clamp(x, 0.0f, 1.0f); }
    template <bool fused = false>
    __device__ float backward(float x) const {
        return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f;
    }
};

struct SqrClippedReLU {
    __device__ float forward(float x) const {
        x = clamp(x, 0.0f, 1.0f);
        return x * x;
    }

    template <bool fused = false>
    __device__ float backward(float x) const {
        if (x > 0.0f && x < 1.0f) {
            if constexpr (fused)
                x = sqrtf(x);
            return 2.0f * x;
        } else {
            return 0.0f;
        }
    }
};

struct Sigmoid {
    __device__ float forward(float x) const { return 1.0f / (1.0f + expf(-x)); }

    template <bool fused = false>
    __device__ float backward(float x) const {
        if constexpr (!fused)
            x = forward(x);
        return x * (1.0f - x);
    }
};

template <typename Op>
struct ElemwiseUnary {
    static constexpr int BLOCK_SIZE = 1024;

    static void forward(const DenseMatrix& in, DenseMatrix& out);
    static void backward(Tensor& in, const DenseMatrix& out_g);
};

using ActOp = std::variant<Linear, ReLU, ClippedReLU, SqrClippedReLU, Sigmoid>;

} // namespace kernel
