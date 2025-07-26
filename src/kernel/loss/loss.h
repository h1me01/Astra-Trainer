#pragma once

#include "../../nn/data/include.h"
#include "../activation.h"

void mpe_loss( //
    const Array<float> &targets,
    Array<float> &loss,
    Tensor<float> &output,
    const float power,
    const ActivationType act_type);

void mse_loss( //
    const Array<float> &targets,
    Array<float> &loss,
    Tensor<float> &output,
    const ActivationType act_type);
