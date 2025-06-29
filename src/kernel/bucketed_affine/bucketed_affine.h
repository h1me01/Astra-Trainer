#pragma once

#include "../../nn/data.h"
#include "../activation.h"

void bucketed_affine_fwd( //
    DenseMatrix &weights_v,
    DenseMatrix &biases_v,
    DenseMatrix &inputs_v,
    DenseMatrix &activated_v,
    DenseMatrix &prev_activated,
    const Array<int> &bucket_indices,
    const ActivationType act_type,
    const int bucket_size);

void bucketed_affine_bwd( //
    Tensor &weights,
    Tensor &biases,
    Tensor &inputs,
    Tensor &activated,
    DenseMatrix &prev_activated,
    const Array<int> &bucket_indices,
    const ActivationType act_type,
    const int bucket_size);