#pragma once

#include "loss.h"

namespace nn {

struct MSE : public Loss {
    void compute(const Array<float>& targets, OpTensor& output) {
        kernel::mse_loss(targets, loss, output.get_output(), output.get_gradients(), act_type);
    }
};

} // namespace nn
