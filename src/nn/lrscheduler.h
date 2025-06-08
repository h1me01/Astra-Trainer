#pragma once

#include "../misc.h"

struct LRScheduler {
    virtual float getLR(int epoch, float lr) = 0;
    virtual std::string getInfo() = 0;
};

struct StepDecay : public LRScheduler {
    int step;
    float gamma;

    StepDecay(int step = 100, float gamma = 1.0) : step(step), gamma(gamma) {}

    float getLR(int epoch, float lr) override {
        return epoch % step == 0 ? lr * gamma : lr;
    }

    std::string getInfo() override {
        return "StepDecay(gamma=" + formatNumber(gamma) + ", step=" + std::to_string(step) + ")";
    }
};

struct GradualDecay : public LRScheduler {
    float gamma;

    GradualDecay(float gamma = 0.92) : gamma(gamma) {}

    float getLR(int epoch, float lr) override {
        return lr * gamma;
    }

    std::string getInfo() override {
        return "GradualDecay(gamma=" + formatNumber(gamma) + ")";
    }
};

class CosineAnnealing : public LRScheduler {
  private:
    int max_epochs;
    float min_lr;
    float initial_lr;
    bool initial_lr_set = false;

    const float pi = 3.14159265358979f;

  public:
    CosineAnnealing(int max_epochs, float initial_lr, float min_lr)
        : max_epochs(max_epochs), initial_lr(initial_lr), min_lr(min_lr) {}

    float getLR(int epoch, float lr) override {
        if(epoch >= max_epochs)
            return min_lr;

        float lambda = 1.0f - 0.5f * (1.0f + std::cos(pi * epoch / max_epochs));
        return initial_lr + lambda * (min_lr - initial_lr);
    }

    std::string getInfo() override {
        return "CosineAnnealing(T_max=" + std::to_string(max_epochs) + ", eta_min=" + formatNumber(min_lr) + ")";
    }
};