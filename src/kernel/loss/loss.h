#pragma once

#include "../../data/include.h"
#include "../elemwise/unary/unary.h"
#include "../util.h"

namespace kernel {

void mpe_loss(const Array<float>& targets, Array<float>& loss, Tensor& out, const float power, ActOp op);

} // namespace kernel
