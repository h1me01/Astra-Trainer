#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

void concat_fwd(const DenseMatrix& in1_v, const DenseMatrix& in2_v, DenseMatrix& out_v, const Activation act_type);

void concat_bwd(DenseMatrix& in1_g, DenseMatrix& in2_g, const Tensor& out, const Activation act_type);

} // namespace kernel
