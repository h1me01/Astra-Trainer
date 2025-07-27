#pragma once

#include "../../nn/data/include.h"
#include "../activation.h"

extern cublasHandle_t CUBLAS_HANDLE;

void create_cublas();
void destroy_cublas();

void affine_fwd( //
    DenseMatrix<float> &weights_v,
    DenseMatrix<float> &biases_v,
    DenseMatrix<float> &inputs_v,
    DenseMatrix<float> &activated_v,
    DenseMatrix<float> &pre_activated,
    const ActivationType act_type);

void affine_bwd( //
    Tensor<float> &weights,
    Tensor<float> &biases,
    Tensor<float> &inputs,
    Tensor<float> &activated,
    DenseMatrix<float> &pre_activated,
    const ActivationType act_type);
