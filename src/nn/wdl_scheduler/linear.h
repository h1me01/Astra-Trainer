#pragma once

#include "wdl_scheduler.h"

namespace nn::wdl_sched {

class Linear : public WDLScheduler {
  public:
    Linear(float start, float final, int max_epochs)
        : WDLScheduler(start),
          start(start),
          final(final),
          max_epochs(max_epochs) {}

    void step(int epoch) override { val = start + (final - start) * (epoch / float(max_epochs)); }

    std::string get_info() const override {
        return "Linear(start=" + format_number(start) + //
               ", final=" + format_number(final) +      //
               ", max_epochs=" + std::to_string(max_epochs) + ")";
    }

  private:
    float start, final;
    int max_epochs;
};

} // namespace nn::wdl_sched
