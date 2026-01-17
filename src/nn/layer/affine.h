#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "layer.h"

namespace nn {

class Affine : public Layer {
  public:
    explicit Affine(int input_size, int output_size, WeightInitType init_type = WeightInitType::He)
        : Layer(input_size, output_size, init_type) {}

    explicit Affine(Affine& other, const Ptr<Layer>& input) {
        this->main = other.get_main();
        ASSERT(this->main != nullptr);
        this->input = input;
        this->input_size = other.input_size;
        this->output_size = other.output_size;
    }

    Ptr<Affine> forward(const Ptr<Layer>& input) {
        if (input == nullptr)
            error("Affine: Input layer is null!");
        if (!is_main)
            error("Affine: Forward can only be used by user defined layers!");
        return std::make_shared<Affine>(*this, input);
    }

    void forward() override {
        ASSERT(!is_main);

        kernel::affine_fwd(
            get_weights().get_values(),
            get_biases().get_values(),
            input->get_output(),
            output.get_linear_output(),
            output.get_activated(),
            act_type
        );
    }

    void backward() override {
        ASSERT(!is_main);

        kernel::affine_bwd(
            get_weights(),
            get_biases(),
            input->get_output(),
            input->get_gradients(),
            output.get_linear_output(),
            output.get_gradients(),
            act_type
        );
    }

    std::vector<Ptr<Layer>> get_inputs() override {
        return {input};
    }

  private:
    Ptr<Layer> input;
};

} // namespace nn
