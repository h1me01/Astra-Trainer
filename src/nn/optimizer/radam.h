#pragma once

#include "optimizer.h"

class RAdam : public Optimizer {
  private:
    int N_sma_threshold = 5;

  public:
    RAdam(AdamParams params = AdamParams{0.001, 0.9, 0.999, 1e-8, 0.01}) : Optimizer(params) {
        name = "RAdam";
    }

    void apply(int batch_size) override {
        step++;

        const float grad_scale = 1.0f / batch_size;

        for(size_t i = 0; i < tunables.size(); i++) {
            radam_optim( //
                tunables[i]->get_data(),
                tunables[i]->get_grads(),
                momentum[i],
                velocity[i],
                params,
                tunables[i]->lower_bound(),
                tunables[i]->upper_bound(),
                grad_scale,
                N_sma_threshold,
                step);
        }
    }

    void setN_SMA_Threshold(int N_sma_threshold) {
        this->N_sma_threshold = N_sma_threshold;
    }
};
