#pragma once

#include "lr_scheduler.h"

namespace nn::lr_sched {

class StepDecay : public LRScheduler {
  public:
    StepDecay(float lr, float gamma, int step_size)
        : LRScheduler(lr),
          base_lr_(lr),
          gamma_(gamma),
          step_size_(step_size) {

        if (lr <= 0.0f)
            error("Step Decay LR Scheduler: initial lr must be positive!");
        if (gamma <= 0.0f)
            error("Step Decay LR Scheduler: gamma must be positive!");
        if (step_size <= 0)
            error("Step Decay LR Scheduler: step_size must be positive!");
    }

    void step(int epoch) override { lr_ = base_lr_ * std::pow(gamma_, (epoch + 1) / step_size_); }

    std::string info() const override {
        return "StepDecay(lr=" + std::to_string(lr_) + //
               ", gamma=" + format_number(gamma_) +    //
               ", step_size=" + std::to_string(step_size_) + ")";
    }

  private:
    float base_lr_;
    float gamma_;
    int step_size_;
};

} // namespace nn::lr_sched
