#pragma once

#include "lr_scheduler.h"

namespace nn::lr_sched {

class CosineAnnealing : public LRScheduler {
  public:
    CosineAnnealing(float start, float final, int max_epochs)
        : LRScheduler(start),
          start(start),
          final(final),
          max_epochs(max_epochs) {

        if (start <= 0)
            error("Cosine Annealing LR Scheduler: start lr must be positive!");
        if (final <= 0)
            error("Cosine Annealing LR Scheduler: final lr must be positive!");
        if (final > start)
            error("Cosine Annealing LR Scheduler: final lr cannot be greater than start lr rate!");
        if (max_epochs <= 0)
            error("Cosine Annealing LR Scheduler: max_epochs must be positive!");
    }

    void step(int epoch) override {
        if (epoch >= max_epochs)
            return;

        float lambda = 1.0f - 0.5f * (1.0f + std::cos(pi * epoch / max_epochs));
        lr = start + lambda * (final - start);
    }

    std::string get_info() const override {
        return "CosineAnnealing(start=" + format_number(start) + //
               ", final=" + format_number(final) +               //
               ", max_epochs=" + std::to_string(max_epochs) + ")";
    }

  private:
    float start;
    float final;
    int max_epochs;

    const float pi = 3.14159265358979f;
};

} // namespace nn::lr_sched
