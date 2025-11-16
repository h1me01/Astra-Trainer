#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

void mpe_loss( //
    const Array<float> &targets,
    Array<float> &loss,
    Tensor<float> &out,
    const float power,
    const ActivationType act_type);

void mse_loss( //
    const Array<float> &targets,
    Array<float> &loss,
    Tensor<float> &out,
    const ActivationType act_type);

} // namespace kernel
