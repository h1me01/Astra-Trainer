#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

// fairly specific fusion for typical multi-layer models

void sparse_affine_pairwise_mul_fwd(
    const DenseMatrix& weights_d,
    const DenseMatrix& biases_d,
    DenseMatrix& out_d,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
);

void sparse_affine_pairwise_mul_bwd(
    Tensor& weights,
    Tensor& biases,
    const Tensor& out,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
);

} // namespace kernel