#pragma once

#include "lr_scheduler.h"
#include <numbers>

namespace nn::lr_sched {

class CosineAnnealing : public LRScheduler {
  public:
    CosineAnnealing(float start, float final, int max_epochs)
        : LRScheduler(start),
          start_(start),
          final_(final),
          max_epochs_(max_epochs - 1) {

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
        if (epoch > max_epochs_)
            return;

        float t = static_cast<float>(epoch) / max_epochs_;
        float lambda = 0.5f * (1.0f - std::cos(std::numbers::pi_v<float> * t));
        lr_ = start_ + lambda * (final_ - start_);
    }

    std::string get_info() const override {
        return "CosineAnnealing(start=" + format_number(start_) + //
               ", final=" + format_number(final_) +               //
               ", max_epochs=" + std::to_string(max_epochs_ + 1) + ")";
    }

  private:
    float start_;
    float final_;
    int max_epochs_;
};

} // namespace nn::lr_sched
