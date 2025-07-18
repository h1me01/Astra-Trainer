#pragma once

#include "../../kernel/include.h"
#include "loss.h"

template <ActivationType act_type> //
struct MSELoss : Loss {
    MSELoss() : Loss() {}

    void compute(const Array<float> &targets, Tensor &output) {
        mse_loss(targets, loss, output, act_type);
    }

    std::string get_info() {
        return "MSELoss<" + get_activation_name(act_type) + ">()";
    }
};
