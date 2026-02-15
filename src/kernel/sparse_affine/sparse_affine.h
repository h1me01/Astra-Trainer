#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

void sparse_affine_fwd(
    const DenseMatrix& weights_d,
    const DenseMatrix& biases_d,
    DenseMatrix& out_d,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
);

void sparse_affine_bwd(
    DenseMatrix& weights_g,
    DenseMatrix& biases_g,
    const Tensor& out,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
);

} // namespace kernel