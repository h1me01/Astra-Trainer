#pragma once

#include "loss.h"

namespace nn::loss {

class MPE : public Loss {
  public:
    MPE(float power, ActivationType act_type)
        : Loss(act_type),
          power(power) {}

    void compute(Tensor& output) {
        kernel::mpe_loss(targets, loss, output, power, act_type);
    }

  private:
    float power;
};

} // namespace nn::loss
