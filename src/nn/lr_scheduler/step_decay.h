#pragma once

#include "lr_scheduler.h"

namespace nn::lr_sched {

class StepDecay : public LRScheduler {
  public:
    StepDecay(float lr, float gamma, int step_size)
        : LRScheduler(lr),
          gamma(gamma),
          step_size(step_size) {}

    void step(int epoch) override {
        if (epoch % step_size == 0)
            lr *= gamma;
    }

    std::string get_info() const override {
        return "StepDecay(lr=" + std::to_string(lr) + //
               ", gamma=" + format_number(gamma) +    //
               ", step_size=" + std::to_string(step_size) + ")";
    }

  private:
    float gamma;
    int step_size;
};

} // namespace nn::lr_sched
