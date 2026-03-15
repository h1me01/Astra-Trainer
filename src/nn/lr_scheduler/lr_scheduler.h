#pragma once

#include "../../misc.h"

namespace nn::lr_sched {

class LRScheduler {
  public:
    LRScheduler(float lr)
        : lr_(lr) {}

    virtual ~LRScheduler() = default;

    virtual void step(int epoch) = 0;

    float get() const { return lr_; }

    virtual std::string get_info() const = 0;

  protected:
    float lr_;
};

} // namespace nn::lr_sched
