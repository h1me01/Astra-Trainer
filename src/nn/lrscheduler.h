#pragma once

#include "../misc.h"

class LRScheduler {
  public:
    virtual float get_lr(int epoch, float lr) = 0;
    virtual std::string get_info() = 0;
};

class StepDecay : public LRScheduler {
  private:
    int m_step;
    float m_gamma;

  public:
    StepDecay(int step = 100, float gamma = 1.0) : m_step(step), m_gamma(gamma) {}

    float get_lr(int epoch, float lr) override {
        return epoch % m_step == 0 ? lr * m_gamma : lr;
    }

    std::string get_info() override {
        return "StepDecay(gamma=" + format_number(m_gamma) + ", step=" + std::to_string(m_step) + ")";
    }
};

class GradualDecay : public LRScheduler {
  private:
    float m_gamma;

  public:
    GradualDecay(float gamma = 0.92) : m_gamma(gamma) {}

    float get_lr(int epoch, float lr) override {
        return lr * m_gamma;
    }

    std::string get_info() override {
        return "GradualDecay(gamma=" + format_number(m_gamma) + ")";
    }
};

class CosineAnnealing : public LRScheduler {
  private:
    int m_max_epochs;
    float m_min_lr;
    float m_initial_lr;

    const float m_pi = 3.14159265358979f;

  public:
    CosineAnnealing(int max_epochs, float initial_lr, float min_lr)
        : m_max_epochs(max_epochs), m_initial_lr(initial_lr), m_min_lr(min_lr) {}

    float get_lr(int epoch, float lr) override {
        if(epoch >= m_max_epochs)
            return m_min_lr;

        float lambda = 1.0f - 0.5f * (1.0f + std::cos(m_pi * epoch / m_max_epochs));
        return m_initial_lr + lambda * (m_min_lr - m_initial_lr);
    }

    std::string get_info() override {
        return "CosineAnnealing(T_max=" + std::to_string(m_max_epochs) + ", eta_min=" + format_number(m_min_lr) + ")";
    }
};