#pragma once

#include "../../data/include.h"
#include "../util.h"

namespace kernel {

void adam_optim(
    Tensor& param,
    Array<float>& moms,
    Array<float>& vels,
    const float lr,
    const float beta1,
    const float beta2,
    const float decay,
    const float grad_scale
);

} // namespace kernel
