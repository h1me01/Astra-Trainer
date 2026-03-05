#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

void pairwise_mul_fwd(const DenseMatrix& in_d, DenseMatrix& out_d, const ActivationType act_type);
void pairwise_mul_bwd(Tensor& in, const Tensor& out, const ActivationType act_type);

} // namespace kernel
