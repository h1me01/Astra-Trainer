#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

void pairwise_mul_fwd(const DenseMatrix& in_d, DenseMatrix& out_d);
void pairwise_mul_bwd(Tensor& in, const Tensor& out);

} // namespace kernel
