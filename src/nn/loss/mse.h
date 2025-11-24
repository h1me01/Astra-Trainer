#pragma once

#include "loss.h"

namespace nn {

struct MSE : public Loss {
    MSE(ActivationType act_type) : Loss(act_type) {}

    void compute(const Array<float> &targets, LayerTensor &output) {
        kernel::mse_loss( //
            targets,
            loss,
            output.get_output(),
            output.get_gradients(),
            act_type);
    }
};

} // namespace nn
