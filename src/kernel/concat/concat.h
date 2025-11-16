#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

void concat_fwd( //
    const DenseMatrix<float> &in1_v,
    const DenseMatrix<float> &in2_v,
    DenseMatrix<float> &out_v);

void concat_bwd( //
    DenseMatrix<float> &in1_g,
    DenseMatrix<float> &in2_g,
    const DenseMatrix<float> &out_g);

} // namespace kernel
