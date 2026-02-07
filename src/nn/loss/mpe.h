#pragma once

#include "loss.h"

namespace nn {

class MPE : public Loss {
  public:
    MPE(float power)
        : power(power) {}

    void compute(const Array<float>& targets, Tensor& output) {
        kernel::mpe_loss(targets, loss, output.get_data(), output.get_grads(), power, act_type);
    }

  private:
    float power;
};

} // namespace nn
