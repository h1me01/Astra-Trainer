#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

extern cublasHandle_t CUBLAS_HANDLE;

void create_cublas();
void destroy_cublas();

// assumes column-major storage
void affine_fwd( //
    DenseMatrix<float> &weights_v,
    DenseMatrix<float> &biases_v,
    DenseMatrix<float> &inputs_v,
    DenseMatrix<float> &out_v);

// assumes column-major storage
void affine_bwd( //
    Tensor<float> &weights,
    Tensor<float> &biases,
    Tensor<float> &inputs,
    Tensor<float> &out);

} // namespace kernel
