#pragma once

#include "../../nn/data.h"
#include "../activation.h"

void createCublas();
void destroyCublas();

// clang-format off
void affine
(
    DenseMatrix &weights_v, 
    DenseMatrix &biases_v, 
    DenseMatrix &inputs_v, 
    DenseMatrix &output_v,
    const ActivationType act_type
);

void affine_bp
(
    Tensor &weights, 
    Tensor &biases, 
    Tensor &inputs, 
    Tensor &output, 
    const ActivationType act_type
);
// clang-format on
