#pragma once

#include "optimizer.h"

namespace nn {

class RAdam : public Optimizer {
  public:
    RAdam(float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8, float decay = 0.01)
        : beta1(beta1),
          beta2(beta2),
          epsilon(epsilon),
          decay(decay),
          curr_step(0) {}

    void step(float lr, int batch_size) override {
        curr_step++;

        const float grad_scale = 1.0f / batch_size;
        const auto radam_params = kernel::optim_utils::get_radam_params(beta1, beta2, curr_step);

        for (size_t i = 0; i < params.size(); i++) {
            kernel::radam_optim(
                *params[i],
                momentum[i],
                velocity[i],
                lr,
                beta1,
                beta2,
                epsilon,
                decay,
                grad_scale,
                radam_params,
                curr_step
            );
        }
    }

  private:
    int curr_step;

    float beta1, beta2;
    float epsilon;
    float decay;
};

} // namespace nn
