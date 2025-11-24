#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

void mpe_loss( //
    const Array<float> &targets,
    Array<float> &loss,
    const DenseMatrix &out,
    DenseMatrix &grads,
    const float power,
    const ActivationType act_type);

void mse_loss( //
    const Array<float> &targets,
    Array<float> &loss,
    const DenseMatrix &out,
    DenseMatrix &grads,
    const ActivationType act_type);

} // namespace kernel
