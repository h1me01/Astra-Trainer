#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

// assumes column-major storage
void feature_transformer_fwd( //
    const DenseMatrix<float> &weights_v,
    const DenseMatrix<float> &biases_v,
    DenseMatrix<float> &out_v,
    const DenseMatrix<int> &features,
    const int max_entries);

// assumes column-major storage
void feature_transformer_bwd( //
    DenseMatrix<float> &weights_g,
    DenseMatrix<float> &biases_g,
    const DenseMatrix<float> &out_g,
    const DenseMatrix<int> &features,
    const int max_entries);

} // namespace kernel
