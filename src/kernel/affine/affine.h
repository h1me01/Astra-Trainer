#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

extern cublasHandle_t CUBLAS_HANDLE;

void create_cublas();
void destroy_cublas();

void affine_fwd( //
    DenseMatrix &weights_v,
    DenseMatrix &biases_v,
    DenseMatrix &inputs_v,
    DenseMatrix &linear_out,
    DenseMatrix &activated,
    ActivationType act_type);

void affine_bwd( //
    Tensor &weights,
    Tensor &biases,
    DenseMatrix &in_v,
    DenseMatrix &in_g,
    DenseMatrix &linear_out,
    DenseMatrix &grads,
    ActivationType act_type);

} // namespace kernel
