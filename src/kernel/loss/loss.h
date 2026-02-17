#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

void mpe_loss(
    const Array<float>& targets, Array<float>& loss, Tensor& out, const float power, const Activation act_type
);

} // namespace kernel
