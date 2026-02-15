#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

void pairwise_mul_fwd(const DenseMatrix& in_d, DenseMatrix& out_d, const int out_offset, const Activation act_type);

void pairwise_mul_bwd(Tensor& in, const Tensor& out, const int out_offset, const Activation act_type);

} // namespace kernel
