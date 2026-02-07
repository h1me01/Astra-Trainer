#pragma once

#include "loss.h"

namespace nn {

class MPE : public Loss {
  public:
    MPE(float power)
        : power(power) {}

    void compute(const Array<float>& targets, OpTensor& output) {
        kernel::mpe_loss(targets, loss, output.get_output(), output.get_gradients(), power, act_type);
    }

  private:
    float power;
};

} // namespace nn
