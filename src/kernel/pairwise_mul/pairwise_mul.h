#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

void pairwise_mul_fwd( //
    const DenseMatrix &in_v,
    DenseMatrix &out_v,
    const int out_offset);

void pairwise_mul_bwd( //
    Tensor &in,
    const DenseMatrix &out_g,
    const int out_offset);

} // namespace kernel
