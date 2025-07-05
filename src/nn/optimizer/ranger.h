#pragma once

#include "optimizer.h"

class Ranger : public Optimizer {
  private:
    float alpha = 0.5;
    int k = 6;
    int N_sma_threshold = 6;

  public:
    Ranger(OptimParams params = OptimParams{0.001, 0.95, 0.999, 1e-5, 0.01}) : Optimizer(params) {
        name = "Ranger";
    }

    void apply(int batch_size) override {
        step++;

        const float grad_scale = 1.0f / batch_size;

        for(size_t i = 0; i < tunables.size(); i++) {
            ranger_optim( //
                tunables[i]->get_data(),
                tunables[i]->get_grads(),
                momentum[i],
                velocity[i],
                slow_buffer[i],
                params,
                tunables[i]->lower_bound(),
                tunables[i]->upper_bound(),
                grad_scale,
                alpha,
                k,
                N_sma_threshold,
                step);
        }
    }

    void setAlpha(float alpha) {
        this->alpha = alpha;
    }

    void setK(int k) {
        this->k = k;
    }

    void setN_SMA_Threshold(int N_sma_threshold) {
        this->N_sma_threshold = N_sma_threshold;
    }
};
