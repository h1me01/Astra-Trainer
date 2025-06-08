#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "../../kernel/kernel.h"
#include "../../misc.h"
#include "../data.h"
#include "layer.h"

template <int size, ActivationType act_type> //
class FullyConnected : public LayerBase {
  private:
    Tensor weights{size, 1};
    Tensor biases{size, 1};

    LayerBase *previous;

  public:
    FullyConnected(LayerBase *previous, bool init_uniformly = true) {
        this->previous = previous;
        int input_size = previous->getOutputSize();

        weights = Tensor(size, input_size);
        if(init_uniformly)
            weights.initUniformly();
        else
            weights.heInit(input_size);
    }

    void forward() override {
        Tensor &inputs = previous->getDenseOutput();
        affine(weights.getValues(), biases.getValues(), inputs.getValues(), dense_output.getValues(), act_type);
    }

    void backprop() override {
        Tensor &inputs = previous->getDenseOutput();
        affine_bp(weights, biases, inputs, dense_output, act_type);
    }

    int getOutputSize() const override {
        return size;
    }

    int getInputSize() const override {
        return previous->getOutputSize();
    }

    std::vector<Tensor *> getTunables() override {
        return {&weights, &biases};
    }

    std::string getInfo() override {
        const std::string activationStrings[] = {"Linear", "ReLU", "CReLU", "SCReLU", "Sigmoid"};

        std::stringstream info;
        info << "FullyConnected(";
        info << "input_size=" << std::to_string(getInputSize());
        info << ", output_size=" << std::to_string(size);
        info << ", activation=" << activationStrings[act_type] << ")\n";
        return info.str();
    }
};
