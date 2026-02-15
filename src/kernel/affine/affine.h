#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

extern cublasHandle_t CUBLAS_HANDLE;

void create_cublas();
void destroy_cublas();

void affine_fwd(
    DenseMatrix& weights_d,
    DenseMatrix& biases_d,
    const DenseMatrix& inputs_d,
    DenseMatrix& out_d,
    const Activation act_type
);
void affine_bwd(Tensor& weights, Tensor& biases, Tensor& in, Tensor& out, const Activation act_type);

} // namespace kernel
