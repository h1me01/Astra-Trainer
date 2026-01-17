#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

void concat_fwd(
    const DenseMatrix& in1_v,
    const DenseMatrix& in2_v,
    DenseMatrix& linear_out,
    DenseMatrix& activated,
    const Activation act_type
);

void concat_bwd(
    DenseMatrix& in1_g,
    DenseMatrix& in2_g,
    const DenseMatrix& linear_out,
    const DenseMatrix& grads,
    const Activation act_type
);

} // namespace kernel
