#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

void select_fwd(const DenseMatrix& in_v, DenseMatrix& out_d, const Array<int>& indices, const Activation act_type);

void select_bwd(DenseMatrix& in_g, const Tensor& out, const Array<int>& indices, const Activation act_type);

} // namespace kernel
