#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

// assumes column-major storage
void feature_transformer_fwd( //
    const DenseMatrix &weights_v,
    const DenseMatrix &biases_v,
    DenseMatrix &out_v,
    DenseMatrix *act_v,
    const Array<int> &features,
    const int max_entries,
    const int out_offset,
    const ActivationType act_type = ActivationType::Linear);

// assumes column-major storage
void feature_transformer_bwd( //
    DenseMatrix &weights_g,
    DenseMatrix &biases_g,
    const DenseMatrix &incoming_grad,
    const DenseMatrix *fwd_out_v,
    const Array<int> &features,
    const int max_entries,
    const int out_offset,
    const ActivationType act_type = ActivationType::Linear);

} // namespace kernel
