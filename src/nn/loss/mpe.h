#pragma once

#include "loss.h"

namespace nn::loss {

class MPE : public Loss {
  public:
    MPE(float power, graph::OpType act_type)
        : Loss(act_type),
          power_(power) {}

    void compute(Tensor& output, const Array<float>& targets) override {
        kernel::mpe_loss(targets, loss_, output, power_, act_op_);
    }

  private:
    float power_;
};

} // namespace nn::loss
