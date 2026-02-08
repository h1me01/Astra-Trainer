#pragma once

#include "loss.h"

namespace nn {

struct MSE : public Loss {
    MSE(Activation act)
        : Loss(act) {}

    void compute(const Array<float>& targets, Tensor& output) {
        kernel::mse_loss(targets, loss, output.get_data(), output.get_grads(), act_type);
    }
};

} // namespace nn
