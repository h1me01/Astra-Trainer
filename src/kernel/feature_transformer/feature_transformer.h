#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

// assumes column-major storage
void feature_transformer_fwd( //
    const DenseMatrix &weights_v,
    const DenseMatrix &biases_v,
    DenseMatrix &out_v,
    const Array<int> &features,
    const int max_entries,
    const int offset);

// assumes column-major storage
void feature_transformer_bwd( //
    DenseMatrix &weights_g,
    DenseMatrix &biases_g,
    const DenseMatrix &out_g,
    const Array<int> &features,
    const int max_entries,
    const int offset);

} // namespace kernel
