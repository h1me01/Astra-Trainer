#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

void pairwise_mul_fwd(const DenseMatrix &in_v, DenseMatrix &out_v);

void pairwise_mul_bwd(Tensor &in, const DenseMatrix &out_g);

} // namespace kernel
