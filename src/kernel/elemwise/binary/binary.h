#pragma once

#include "../../../data/include.h"
#include "../../util.h"

namespace kernel {

struct AddBinary {
    __device__ float forward(float a, float b) const { return a + b; }
    __device__ void backward(float go, float a, float b, float& ga, float& gb) const {
        ga += go;
        gb += go;
    }
};

struct SubBinary {
    __device__ float forward(float a, float b) const { return a - b; }
    __device__ void backward(float go, float a, float b, float& ga, float& gb) const {
        ga += go;
        gb -= go;
    }
};

struct MulBinary {
    __device__ float forward(float a, float b) const { return a * b; }
    __device__ void backward(float go, float a, float b, float& ga, float& gb) const {
        ga += go * b;
        gb += go * a;
    }
};

struct DivBinary {
    __device__ float forward(float a, float b) const { return a / b; }
    __device__ void backward(float go, float a, float b, float& ga, float& gb) const {
        ga += go / b;
        gb += -go * a / (b * b);
    }
};

template <typename Op>
struct ElemwiseBinary {
    static constexpr int BLOCK_SIZE = 1024;

    static void forward(const DenseMatrix& a, const DenseMatrix& b, DenseMatrix& c, Op op);
    static void backward(Tensor& a, Tensor& b, const DenseMatrix& grad_out, Op op);
};

} // namespace kernel
