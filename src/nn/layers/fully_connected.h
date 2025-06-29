#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "../../kernel/kernel.h"
#include "../../misc.h"
#include "../data.h"
#include "layer.h"

template <int size, ActivationType act_type = Linear> //
class FullyConnected : public LayerBase {
  private:
    Tensor weights{size, 1};
    Tensor biases{size, 1};

    LayerBase *previous;

  public:
    FullyConnected(LayerBase *previous, bool init_uniformly = true) {
        name = "FullyConnected";

        this->previous = previous;
        int input_size = previous->getOutputSize();

        weights = Tensor(size, input_size);
        if(init_uniformly)
            weights.initUniformly();
        else
            weights.heInit(input_size);
    }

    void forward() override {
        Tensor &inputs = previous->getDenseOutput().activated;
        affine( //
            weights.getValues(),
            biases.getValues(),
            inputs.getValues(),
            output.activated.getValues(),
            output.pre_activated,
            act_type);
    }

    void backprop() override {
        Tensor &inputs = previous->getDenseOutput().activated;
        affine_bp( //
            weights,
            biases,
            inputs,
            output.activated,
            output.pre_activated,
            act_type);
    }

    ActivationType getActivationType() const override {
        return act_type;
    }

    int getOutputSize() const override {
        return size;
    }

    int getInputSize() const override {
        return previous->getOutputSize();
    }

    std::vector<Tensor *> getParams() override {
        return {&weights, &biases};
    }
};
