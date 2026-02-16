#pragma once

#include "../../misc.h"

namespace nn::lr_sched {

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

    float get() const { return lr; }

    virtual std::string get_info() const = 0;

  protected:
    float lr;
};

} // namespace nn::lr_sched
