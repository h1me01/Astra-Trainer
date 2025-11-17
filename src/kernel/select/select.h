#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

void select_fwd( //
    const DenseMatrix &in_v,
    DenseMatrix &out_v,
    const Array<int> &indices);

void select_bwd( //
    DenseMatrix &in_g,
    const DenseMatrix &out_g,
    const Array<int> &indices);

} // namespace kernel
