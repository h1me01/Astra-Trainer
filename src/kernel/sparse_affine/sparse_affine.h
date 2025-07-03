#pragma once

#include "../../nn/data.h"
#include "../activation.h"

void sparse_affine_fwd( //
    DenseMatrix &activated_v,
    DenseMatrix &pre_activated,
    const DenseMatrix &weights_v,
    const DenseMatrix &biases_v,
    const Array<int> &features,
    const Array<int> &feature_sizes,
    const int a_offset,
    const int batch_size,
    const int max_entries,
    const ActivationType act_type);

void sparse_affine_bwd( //
    const DenseMatrix &activated_g,
    const DenseMatrix &pre_activated,
    DenseMatrix &weights_g,
    DenseMatrix &biases_g,
    const Array<int> &features,
    const Array<int> &feature_sizes,
    const int a_offset,
    const int batch_size,
    const int max_entries,
    const ActivationType act_type);
