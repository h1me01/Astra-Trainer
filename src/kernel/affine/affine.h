#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

extern cublasHandle_t CUBLAS_HANDLE;

void create_cublas();
void destroy_cublas();

void affine_fwd(
    DenseMatrix& weights_v, DenseMatrix& biases_v, const DenseMatrix& inputs_v, DenseMatrix& out_d, Activation act_type
);

void affine_bwd(Tensor& weights, Tensor& biases, Tensor& in, Tensor& out, Activation act_type);

} // namespace kernel
