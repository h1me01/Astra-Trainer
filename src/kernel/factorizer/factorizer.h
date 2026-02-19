#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

void factorizer_fwd(
    const DenseMatrix& factorizer_d, const DenseMatrix& weights_d, DenseMatrix& out_d, const int out_offset
);
void factorizer_bwd(DenseMatrix& in_g, const DenseMatrix& out_g, const int out_offset);

} // namespace kernel
