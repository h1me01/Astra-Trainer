#pragma once

#include "layer.h"

namespace nn {

class PairwiseMul : public Layer {
  public:
    explicit PairwiseMul(const Ptr<Layer> &input, const Ptr<Layer> &input2 = nullptr) {
        inputs.push_back(input);
        if(input2)
            inputs.push_back(input2);
    }

    void init(int batch_size) override {
        input_size = inputs[0]->get_output_size();

        if(inputs.size() > 1 && inputs[1]->get_output_size() != input_size)
            error("PairwiseMul: Inputs must have the same size!");
        if(input_size % 2 != 0)
            error("PairwiseMul: Input size must be even!");

        output_size = (input_size / 2) * inputs.size();

        Layer::init(batch_size);
    }

    void forward() override {
        for(int i = 0; i < inputs.size(); i++) {
            kernel::pairwise_mul_fwd( //
                inputs[i]->get_output().get_values(),
                output.get_values(),
                i * (input_size / 2));
        }

        activation.forward(output.get_values());
    }

    void backward() override {
        activation.backward(output);

        for(int i = 0; i < inputs.size(); i++) {
            kernel::pairwise_mul_bwd( //
                inputs[i]->get_output(),
                output.get_gradients(),
                i * (input_size / 2));
        }
    }

    std::vector<Ptr<Layer>> get_inputs() override {
        return inputs;
    }

  private:
    std::vector<Ptr<Layer>> inputs;
};

} // namespace nn
