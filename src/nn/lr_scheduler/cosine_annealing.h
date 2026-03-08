#pragma once

#include "lr_scheduler.h"
#include <numbers>

namespace nn::lr_sched {

class CosineAnnealing : public LRScheduler {
  public:
    CosineAnnealing(float start, float final, int max_epochs)
        : LRScheduler(start),
          start(start),
          final(final),
          max_epochs(max_epochs - 1) {

        if (start <= 0)
            error("Cosine Annealing LR Scheduler: start lr must be positive!");
        if (final <= 0)
            error("Cosine Annealing LR Scheduler: final lr must be positive!");
        if (final > start)
            error("Cosine Annealing LR Scheduler: final lr cannot be greater than start lr rate!");
        if (max_epochs <= 1)
            error("Cosine Annealing LR Scheduler: max_epochs must be greater than 1!");
    }

    void step(int epoch) override {
        if (epoch > max_epochs)
            return;

        float t = static_cast<float>(epoch) / max_epochs;
        float lambda = 0.5f * (1.0f - std::cos(pi * t));
        lr = start + lambda * (final - start);
    }

    std::string get_info() const override {
        return "CosineAnnealing(start=" + format_number(start) + //
               ", final=" + format_number(final) +               //
               ", max_epochs=" + std::to_string(max_epochs + 1) + ")";
    }

  private:
    float start;
    float final;
    int max_epochs;

    const float pi = std::numbers::pi_v<float>;
};

} // namespace nn::lr_sched
