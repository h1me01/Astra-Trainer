#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

// assumes column-major storage
void feature_transformer_fwd(
    const DenseMatrix& weights_v,
    const DenseMatrix& biases_v,
    DenseMatrix& out_v,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
);

// assumes column-major storage
void feature_transformer_bwd(
    DenseMatrix& weights_g,
    DenseMatrix& biases_g,
    const Tensor& out,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
);

} // namespace kernel
