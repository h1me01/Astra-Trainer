#pragma once

#include "lr_scheduler.h"

namespace nn {

class StepDecay : public LRScheduler {
  public:
    StepDecay(float lr, int step_size = 100, float gamma = 0.1) //
        : LRScheduler(lr), step_size(step_size), gamma(gamma) {}

    void step(int epoch) override {
        lr = (epoch % step_size == 0) ? lr * gamma : lr;
    }

  private:
    int step_size;
    float gamma;
};

} // namespace nn
