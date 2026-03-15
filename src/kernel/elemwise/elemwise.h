#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

struct Add {
    __device__ float forward(float a, float b) const { return a + b; }
    __device__ void backward(float go, float a, float b, float& ga, float& gb) const {
        ga += go;
        gb += go;
    }
};

struct Sub {
    __device__ float forward(float a, float b) const { return a - b; }
    __device__ void backward(float go, float a, float b, float& ga, float& gb) const {
        ga += go;
        gb -= go;
    }
};

struct Mul {
    __device__ float forward(float a, float b) const { return a * b; }
    __device__ void backward(float go, float a, float b, float& ga, float& gb) const {
        ga += go * b;
        gb += go * a;
    }
};

struct Div {
    __device__ float forward(float a, float b) const { return a / b; }
    __device__ void backward(float go, float a, float b, float& ga, float& gb) const {
        ga += go / b;
        gb += -go * a / (b * b);
    }
};

template <typename Op>
struct Elemwise {
    static constexpr int BLOCK_SIZE = 256;

    static void forward(const DenseMatrix& a, const DenseMatrix& b, DenseMatrix& c);
    static void backward(const Tensor& a, const Tensor& b, const DenseMatrix& grad_out);
};

} // namespace kernel
