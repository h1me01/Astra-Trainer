#pragma once

#include "../../kernel/include.h"
#include "loss.h"

namespace nn {

class MPE : public Loss {
  public:
    MPE(ActivationType act_type, float power) //
        : Loss(act_type), power(power) {}

    void compute(const Array<float> &targets, Tensor<float> &output) {
        kernel::mpe_loss(targets, loss, output, power, act_type);
    }

  private:
    float power;
};

} // namespace nn
