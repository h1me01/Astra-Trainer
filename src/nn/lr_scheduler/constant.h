#pragma once

#include "lr_scheduler.h"

namespace nn::lr_sched {

struct Constant : public LRScheduler {
    Constant(float lr)
        : LRScheduler(lr) {
        if (lr <= 0.0f)
            error("Constant LR Scheduler: lr must be positive!");
    }

    void step([[maybe_unused]] int epoch) override {}

    std::string info() const override { return "Constant(lr=" + format_number(lr_) + ")"; }
};

} // namespace nn::lr_sched
