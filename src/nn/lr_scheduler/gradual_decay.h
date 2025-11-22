#pragma once

#include "lr_scheduler.h"

namespace nn {

class GradualDecay : public LRScheduler {
  public:
    GradualDecay(float lr, float gamma = 0.92) //
        : LRScheduler(lr), gamma(gamma) {}

    void step(int epoch) override {
        lr *= gamma;
    }

  private:
    float gamma;
};

} // namespace nn
