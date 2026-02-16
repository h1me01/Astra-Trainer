#pragma once

#include "lr_scheduler.h"

namespace nn::lr_sched {

class Constant : public LRScheduler {
  public:
    Constant(float lr)
        : LRScheduler(lr) {
        if (lr <= 0.0f)
            error("Constant LR Scheduler: lr must be positive!");
    }

    void step([[maybe_unused]] int epoch) override {}

    std::string get_info() const override { return "Constant(lr=" + format_number(lr) + ")"; }
};

} // namespace nn::lr_sched
