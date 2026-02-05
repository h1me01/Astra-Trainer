#pragma once

#include "../../misc.h"

namespace nn {

class LRScheduler {
  public:
    LRScheduler(float lr)
        : lr(lr) {}

    virtual ~LRScheduler() = default;

    virtual void step(int epoch) = 0;

    void lr_from_epoch(int epoch) {
        for (int i = 1; i <= epoch; i++)
            step(i);
    }

    float get_lr() const { return lr; }

  protected:
    float lr;
};

} // namespace nn
