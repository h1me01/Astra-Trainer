#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

void concat_fwd(const DenseMatrix& in_d, DenseMatrix& out_d, const int offset, const Activation act_type);

void concat_bwd(DenseMatrix& in_g, const Tensor& out, const int offset, const Activation act_type);

} // namespace kernel
