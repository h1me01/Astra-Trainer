#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "layer.h"

namespace nn {

class Affine : public Layer {
  public:
    Affine(int input_size, int output_size, WeightInitType init_type = WeightInitType::He) //
        : Layer(input_size, output_size, init_type) {}

    Affine(Affine &other, const Ptr<Layer> &input) {
        this->main = other.get_main();
        ASSERT(this->main != nullptr);
        this->input = input;
        this->input_size = other.input_size;
        this->output_size = other.output_size;
    }

    Ptr<Affine> forward(const Ptr<Layer> &input) {
        if(input == nullptr)
            error("Affine: Input layer is null!");
        if(!is_main)
            error("Affine: Forward can only be used by user defined layers!");
        return std::make_shared<Affine>(*this, input);
    }

    void forward() override {
        if(is_main)
            return; // main layer does not perform forward

        auto params = get_main()->get_params();

        Tensor &weights = *params[0];
        Tensor &biases = *params[1];

        kernel::affine_fwd( //
            weights.get_values(),
            biases.get_values(),
            input->get_output().get_values(),
            output.get_values());

        activation.forward(output.get_values());
    }

    void backward() override {
        if(is_main)
            return; // main layer does not perform backward

        auto params = get_main()->get_params();

        Tensor &weights = *params[0];
        Tensor &biases = *params[1];

        activation.backward(output);

        kernel::affine_bwd( //
            weights,
            biases,
            input->get_output(),
            output);
    }

    std::vector<Ptr<Layer>> get_inputs() override {
        return {input};
    }

  private:
    Ptr<Layer> input;
};

} // namespace nn
