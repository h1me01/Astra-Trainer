#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

void pairwise_mul_fwd( //
    const DenseMatrix& in_v,
    DenseMatrix& linear_out,
    DenseMatrix& activated,
    const int out_offset,
    const Activation act_type
);

void pairwise_mul_bwd( //
    const DenseMatrix& in_v,
    DenseMatrix& in_g,
    const DenseMatrix& linear_out,
    const DenseMatrix& grads,
    const int out_offset,
    const Activation act_type
);

} // namespace kernel
