#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

void select_fwd(const DenseMatrix& in_d, DenseMatrix& out_d, const Array<int>& indices);
void select_bwd(DenseMatrix& in_g, const Tensor& out, const Array<int>& indices);

} // namespace kernel
