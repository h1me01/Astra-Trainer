#pragma once

#include "lr_scheduler.h"

namespace nn::lr_sched {

class StepDecay : public LRScheduler {
  public:
    StepDecay(float lr, float gamma, int step_size)
        : LRScheduler(lr),
          base_lr(lr),
          gamma(gamma),
          step_size(step_size) {

        if (lr <= 0.0f)
            error("Step Decay LR Scheduler: initial lr must be positive!");
        if (gamma <= 0.0f)
            error("Step Decay LR Scheduler: gamma must be positive!");
        if (step_size <= 0)
            error("Step Decay LR Scheduler: step_size must be positive!");
    }

    void step(int epoch) override { lr = base_lr * std::pow(gamma, (epoch + 1) / step_size); }

    std::string get_info() const override {
        return "StepDecay(lr=" + std::to_string(lr) + //
               ", gamma=" + format_number(gamma) +    //
               ", step_size=" + std::to_string(step_size) + ")";
    }

  private:
    float base_lr;
    float gamma;
    int step_size;
};

} // namespace nn::lr_sched
