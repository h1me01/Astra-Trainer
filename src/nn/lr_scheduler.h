#pragma once

#include "../misc.h"

class LRScheduler {
  public:
    virtual float get_lr(int epoch, float lr) = 0;
    virtual std::string get_info() = 0;
};

class StepDecay : public LRScheduler {
  private:
    int step;
    float gamma;

  public:
    StepDecay(int step = 100, float gamma = 1.0) : step(step), gamma(gamma) {}

    float get_lr(int epoch, float lr) override {
        return epoch % step == 0 ? lr * gamma : lr;
    }

    std::string get_info() override {
        return "StepDecay(gamma=" + format_number(gamma) + ", step=" + std::to_string(step) + ")";
    }
};

class GradualDecay : public LRScheduler {
  private:
    float gamma;

  public:
    GradualDecay(float gamma = 0.92) : gamma(gamma) {}

    float get_lr(int epoch, float lr) override {
        return lr * gamma;
    }

    std::string get_info() override {
        return "GradualDecay(gamma=" + format_number(gamma) + ")";
    }
};

class CosineAnnealing : public LRScheduler {
  private:
    int max_epochs;
    float initial_lr;
    float final_lr;

    const float m_pi = 3.14159265358979f;

  public:
    CosineAnnealing(int max_epochs, float initial_lr, float final_lr)
        : max_epochs(max_epochs), initial_lr(initial_lr), final_lr(final_lr) {}

    float get_lr(int epoch, float lr) override {
        if(epoch >= max_epochs)
            return final_lr;

        float lambda = 1.0f - 0.5f * (1.0f + std::cos(m_pi * epoch / max_epochs));
        return initial_lr + lambda * (final_lr - initial_lr);
    }

    std::string get_info() override {
        return "CosineAnnealing(T_max=" + std::to_string(max_epochs) + ", eta_min=" + format_number(final_lr) + ")";
    }
};