#pragma once

#include "wdl_scheduler.h"

namespace nn::wdl_sched {

class Linear : public WDLScheduler {
  public:
    Linear(float start, float final, int max_epochs)
        : WDLScheduler(start),
          start_(start),
          final_(final),
          max_epochs_(max_epochs - 1) {

        if (start < 0 || start > 1 || final < 0 || final > 1)
            error("Linear WDL Scheduler: start value must be between 0 and 1!");
        if (final < 0 || final > 1)
            error("Linear WDL Scheduler: final value must be between 0 and 1!");
        if (max_epochs <= 1)
            error("Linear WDL Scheduler: max_epochs must be greater than 1!");
    }

    void step(int epoch) override {
        if (epoch > max_epochs_)
            return;

        float t = static_cast<float>(epoch) / max_epochs_;
        val_ = start_ + (final_ - start_) * t;
    }

    std::string info() const override {
        return "Linear(start=" + format_number(start_) + //
               ", final=" + format_number(final_) +      //
               ", max_epochs=" + std::to_string(max_epochs_ + 1) + ")";
    }

  private:
    float start_, final_;
    int max_epochs_;
};

} // namespace nn::wdl_sched
