#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "input.h"
#include "layer.h"

namespace nn {

class FeatureTransformer : public LayerBase {
  public:
    FeatureTransformer(int input_size, int output_size, WeightInitType winit_type = WeightInitType::He)
        : LayerBase(input_size, output_size, winit_type) {

        if(input_size % 768 != 0)
            error("Input size must be divisible by 768 to match standard chess inputs!");
    }

    FeatureTransformer(FeatureTransformer &other, const InputPtr &input) {
        this->main = other.get_main();
        ASSERT(this->main != nullptr);
        this->input = input;
        this->input_size = other.input_size;
        this->output_size = other.output_size;
    }

    FeatureTransformerPtr forward(const InputPtr &input) {
        if(input == nullptr)
            error("Feature Transformer: Input layer is null!");
        if(!is_main)
            error("Feature Transformer: Forward can only be used by user defined layers!");
        return std::make_shared<FeatureTransformer>(*this, input);
    }

    void forward() override {
        if(is_main)
            return; // main layer does not perform forward

        auto params = get_main()->get_params();

        Tensor<float> &weights = *params[0];
        Tensor<float> &biases = *params[1];

        kernel::feature_transformer_fwd( //
            weights.get_values(),
            biases.get_values(),
            output.get_values(),
            input->get_output().get_values(),
            input->get_size());

        activation.forward(output.get_values());
    }

    void backward() override {
        if(is_main)
            return; // main layer does not perform backward

        auto params = get_main()->get_params();

        Tensor<float> &weights = *params[0];
        Tensor<float> &biases = *params[1];

        activation.backward(output);

        kernel::feature_transformer_bwd( //
            weights.get_gradients(),
            biases.get_gradients(),
            output.get_gradients(),
            input->get_output().get_values(),
            input->get_size());
    }

    std::vector<LayerPtr> get_inputs() override {
        return {};
    }

  private:
    InputPtr input;
};

} // namespace nn
