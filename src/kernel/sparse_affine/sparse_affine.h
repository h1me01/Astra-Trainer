#pragma once

#include "../../nn/data/include.h"
#include "../activation.h"

void sparse_affine_fwd( //
    DenseMatrix<float> &activated_v,
    DenseMatrix<float> &pre_activated,
    const DenseMatrix<float> &weights_v,
    const DenseMatrix<float> &biases_v,
    const Array<int> &features,
    const Array<int> &feature_sizes,
    const int a_offset,
    const int max_entries,
    const ActivationType act_type);

void sparse_affine_bwd( //
    const DenseMatrix<float> &activated_g,
    const DenseMatrix<float> &pre_activated,
    DenseMatrix<float> &weights_g,
    DenseMatrix<float> &biases_g,
    const Array<int> &features,
    const Array<int> &feature_sizes,
    const int a_offset,
    const int max_entries,
    const ActivationType act_type);
