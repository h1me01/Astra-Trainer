#pragma once

#include "optimizer.h"

namespace nn {

class Adam : public Optimizer {
  public:
    Adam(float beta1 = 0.9,    //
         float beta2 = 0.999,  //
         float epsilon = 1e-8, //
         float decay = 0.01)
        : beta1(beta1), beta2(beta2), epsilon(epsilon), decay(decay) {}

    void step(float lr, int batch_size) override {
        const float grad_scale = 1.0f / batch_size;
        for(size_t i = 0; i < tunables.size(); i++) {
            kernel::adam_optim( //
                *tunables[i],
                momentum[i],
                velocity[i],
                lr,
                beta1,
                beta2,
                epsilon,
                decay,
                grad_scale);
        }
    }

  private:
    float beta1, beta2;
    float epsilon;
    float decay;
};

} // namespace nn
