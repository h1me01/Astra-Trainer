#pragma once

#include "lr_scheduler.h"

namespace nn {

class StepDecay : public LRScheduler {
  public:
    StepDecay(float lr, float gamma = 0.1, int step_size = 100)
        : LRScheduler(lr),
          gamma(gamma),
          step_size(step_size) {}

    void step(int epoch) override {
        if (epoch % step_size == 0)
            lr *= gamma;
    }

  private:
    float gamma;
    int step_size;
};

} // namespace nn
