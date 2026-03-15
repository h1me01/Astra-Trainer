#pragma once

#include "optimizer.h"

namespace nn::optim {

class Adam : public Optimizer {
  public:
    Adam(float beta1 = 0.9, float beta2 = 0.999, float decay = 0.0)
        : beta1_(beta1),
          beta2_(beta2),
          decay_(decay) {

        if (beta1 < 0.0f || beta1 >= 1.0f)
            error("Adam optimizer: beta1 must be in the range [0, 1)!");
        if (beta2 < 0.0f || beta2 >= 1.0f)
            error("Adam optimizer: beta2 must be in the range [0, 1)!");
        if (decay < 0.0f)
            error("Adam optimizer: decay must be non-negative!");
    }

    void step(float lr, int batch_size) override {
        const float grad_scale = 1.0f / batch_size;
        for (size_t i = 0; i < params_.size(); i++)
            kernel::adam_optim(*params_[i], momentum_[i], velocity_[i], lr, beta1_, beta2_, decay_, grad_scale);
    }

  private:
    float beta1_, beta2_, decay_;
};

} // namespace nn::optim
