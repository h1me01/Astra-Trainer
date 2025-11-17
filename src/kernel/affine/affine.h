#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

extern cublasHandle_t CUBLAS_HANDLE;

void create_cublas();
void destroy_cublas();

// assumes column-major storage
void affine_fwd( //
    DenseMatrix &weights_v,
    DenseMatrix &biases_v,
    DenseMatrix &inputs_v,
    DenseMatrix &out_v);

// assumes column-major storage
void affine_bwd( //
    Tensor &weights,
    Tensor &biases,
    Tensor &inputs,
    Tensor &out);

} // namespace kernel
