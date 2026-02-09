#pragma once

#include "optimizer.h"

namespace nn {

class Adam : public Optimizer {
  public:
    Adam(float beta1 = 0.9, float beta2 = 0.999, float decay = 0.0)
        : beta1(beta1),
          beta2(beta2),
          decay(decay) {}

    void step(float lr, int batch_size) override {
        const float grad_scale = 1.0f / batch_size;
        for (size_t i = 0; i < params.size(); i++)
            kernel::adam_optim(*params[i], momentum[i], velocity[i], lr, beta1, beta2, decay, grad_scale);
    }

  private:
    float beta1, beta2, decay;
};

} // namespace nn
