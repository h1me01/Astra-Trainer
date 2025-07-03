#pragma once

#include "../../nn/data.h"
#include "../activation.h"

void create_cublas();
void destroy_cublas();

void affine_fwd( //
    DenseMatrix &weights_v,
    DenseMatrix &biases_v,
    DenseMatrix &inputs_v,
    DenseMatrix &activated_v,
    DenseMatrix &pre_activated,
    const ActivationType act_type);

void affine_bwd( //
    Tensor &weights,
    Tensor &biases,
    Tensor &inputs,
    Tensor &activated,
    DenseMatrix &pre_activated,
    const ActivationType act_type);
