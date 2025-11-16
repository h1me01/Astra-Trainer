#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

void select_fwd( //
    const DenseMatrix<float> &in_v,
    DenseMatrix<float> &out_v,
    const Array<int> &indices);

void select_bwd( //
    DenseMatrix<float> &in_g,
    const DenseMatrix<float> &out_g,
    const Array<int> &indices);

} // namespace kernel
