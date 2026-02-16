#pragma once

#include "loss.h"

namespace nn::loss {

class MPE : public Loss {
  public:
    MPE(float power, Activation act_type)
        : Loss(act_type),
          power(power) {}

    void compute(const Array<float>& targets, Tensor& output) {
        kernel::mpe_loss(targets, loss, output, power, act_type);
    }

  private:
    float power;
};

} // namespace nn::loss
