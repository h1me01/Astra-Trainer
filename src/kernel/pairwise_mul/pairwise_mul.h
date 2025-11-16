#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

void pairwise_mul_fwd(const DenseMatrix<float> &in_v, DenseMatrix<float> &out_v);

void pairwise_mul_bwd(Tensor<float> &in, const DenseMatrix<float> &out_g);

} // namespace kernel
