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
    StepDecay(int step = 100, float gamma = 0.1) : step(step), gamma(gamma) {}

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
    float start_lr;
    float final_lr;

    const float pi = 3.14159265358979f;

  public:
    CosineAnnealing(int max_epochs, float start_lr, float final_lr)
        : max_epochs(max_epochs), start_lr(start_lr), final_lr(final_lr) {}

    float get_lr(int epoch, float lr) override {
        if(epoch >= max_epochs)
            return final_lr;

        float lambda = 1.0f - 0.5f * (1.0f + std::cos(pi * epoch / max_epochs));
        return start_lr + lambda * (final_lr - start_lr);
    }

    std::string get_info() override {
        std::stringstream ss;
        ss << "CosineAnnealing(max_epochs=" << std::to_string(max_epochs);
        ss << ", start_lr=" << format_number(start_lr);
        ss << ", final_lr=" << format_number(final_lr) << ")";
        return ss.str();
    }
};