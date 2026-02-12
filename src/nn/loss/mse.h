#pragma once

#include "loss.h"

namespace nn {

struct MSE : public Loss {
    MSE(Activation act_type)
        : Loss(act_type) {}

    void compute(const Array<float>& targets, Tensor& output) { kernel::mse_loss(targets, loss, output, act_type); }
};

} // namespace nn
