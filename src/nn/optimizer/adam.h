#pragma once

#include "optimizer.h"

namespace nn::optim {

class Adam : public Optimizer {
  public:
    Adam(float beta1 = 0.9, float beta2 = 0.999, float decay = 0.0)
        : beta1(beta1),
          beta2(beta2),
          decay(decay) {

        if (beta1 < 0.0f || beta1 >= 1.0f)
            error("Adam optimizer: beta1 must be in the range [0, 1)!");
        if (beta2 < 0.0f || beta2 >= 1.0f)
            error("Adam optimizer: beta2 must be in the range [0, 1)!");
        if (decay < 0.0f)
            error("Adam optimizer: decay must be non-negative!");
    }

    void step(float lr, int batch_size) override {
        const float grad_scale = 1.0f / batch_size;
        for (size_t i = 0; i < params.size(); i++)
            kernel::adam_optim(*params[i], momentum[i], velocity[i], lr, beta1, beta2, decay, grad_scale);
    }

  private:
    float beta1, beta2, decay;
};

} // namespace nn::optim
