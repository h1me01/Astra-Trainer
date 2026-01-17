#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

void select_fwd(
    const DenseMatrix& in_v,
    DenseMatrix& linear_out,
    DenseMatrix& activated,
    const Array<int>& indices,
    const Activation act_type
);

void select_bwd(
    DenseMatrix& in_g,
    const DenseMatrix& linear_out,
    const DenseMatrix& grads,
    const Array<int>& indices,
    const Activation act_type
);

} // namespace kernel
