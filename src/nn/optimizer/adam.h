#pragma once

#include "optimizer.h"

struct Adam : Optimizer {
    Adam(OptimParams params = OptimParams{0.001, 0.9, 0.999, 1e-8, 0.01}) : Optimizer(params) {
        name = "Adam";
    }

    void step(int batch_size) override {
        const float grad_scale = 1.0f / batch_size;

        for(size_t i = 0; i < tunables.size(); i++) {
            adam_optim( //
                tunables[i]->get_data(),
                tunables[i]->get_grads(),
                momentum[i],
                velocity[i],
                params,
                tunables[i]->lower_bound(),
                tunables[i]->upper_bound(),
                grad_scale);
        }
    }
};
