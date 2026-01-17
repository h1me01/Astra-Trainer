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

        const float beta1_t = std::pow(beta1, curr_step);
        const float beta2_t = std::pow(beta2, curr_step);
        const int N_sma_max = 2 / (1 - beta2) - 1;
        const int N_sma = N_sma_max - 2 * curr_step * beta2_t / (1 - beta2_t);

        for (size_t i = 0; i < tunables.size(); i++) {
            kernel::radam_optim(
                *tunables[i],
                momentum[i],
                velocity[i],
                lr,
                beta1,
                beta2,
                beta1_t,
                beta2_t,
                epsilon,
                decay,
                grad_scale,
                N_sma,
                N_sma_max,
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
