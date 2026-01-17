#pragma once

#include "lr_scheduler.h"

namespace nn {

class CosineAnnealing : public LRScheduler {
  public:
    CosineAnnealing(int max_epochs, float lr, float final_lr)
        : LRScheduler(lr),
          max_epochs(max_epochs),
          start_lr(lr),
          final_lr(final_lr) {}

    void step(int epoch) override {
        if (epoch >= max_epochs)
            return;

        float lambda = 1.0f - 0.5f * (1.0f + std::cos(pi * epoch / max_epochs));
        lr = start_lr + lambda * (final_lr - start_lr);
    }

  private:
    int max_epochs;
    float start_lr;
    float final_lr;

    const float pi = 3.14159265358979f;
};

} // namespace nn
