#pragma once

#include "loss.h"

namespace nn::loss {

class MPE : public Loss {
  public:
    MPE(float power, ActivationType act_type)
        : Loss(act_type),
          power_(power) {}

    void compute(Tensor& output, const Array<float>& targets) override {
        kernel::mpe_loss(targets, loss_, output, power_, act_type_);
    }

  private:
    float power_;
};

} // namespace nn::loss
