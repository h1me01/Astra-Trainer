#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "input.h"
#include "layer.h"

namespace nn {

class FeatureTransformer : public Layer {
  public:
    FeatureTransformer(int input_size, int output_size, WeightInitType winit_type = WeightInitType::He)
        : Layer(input_size, output_size, winit_type) {

        if(input_size % 768 != 0)
            error("Input size must be divisible by 768 to match standard chess inputs!");
    }

    FeatureTransformer(FeatureTransformer &other, const Ptr<Input> &input) {
        this->main = other.get_main();
        this->inputs = {input};
        this->input_size = other.input_size;
        this->output_size = other.output_size;
        ASSERT(main != nullptr);
    }

    FeatureTransformer(FeatureTransformer &other, const Ptr<Input> &input1, const Ptr<Input> &input2)
        : FeatureTransformer(other, input1) {
        this->inputs.push_back(input2);
        this->output_size = 2 * other.output_size;
    }

    Ptr<FeatureTransformer> forward(const Ptr<Input> &input) {
        if(input == nullptr)
            error("Feature Transformer: Input layer is null!");
        if(!is_main)
            error("Feature Transformer: Forward can only be used by user defined layers!");
        return std::make_shared<FeatureTransformer>(*this, input);
    }

    Ptr<FeatureTransformer> forward(const Ptr<Input> &input1, const Ptr<Input> &input2) {
        if(input1 == nullptr || input2 == nullptr)
            error("Feature Transformer: One of the input layers is null!");
        if(!is_main)
            error("Feature Transformer: Forward can only be used by user defined layers!");
        return std::make_shared<FeatureTransformer>(*this, input1, input2);
    }

    void forward() override {
        if(is_main)
            return; // main layer does not perform forward

        auto params = get_main()->get_params();

        Tensor &weights = *params[0];
        Tensor &biases = *params[1];

        for(int i = 0; i < inputs.size(); i++) {
            kernel::feature_transformer_fwd( //
                weights.get_values(),
                biases.get_values(),
                output.get_values(),
                inputs[i]->get_output(),
                inputs[i]->get_size(),
                i);
        }

        activation.forward(output.get_values());
    }

    void backward() override {
        if(is_main)
            return; // main layer does not perform backward

        auto params = get_main()->get_params();

        Tensor &weights = *params[0];
        Tensor &biases = *params[1];

        activation.backward(output);

        for(int i = 0; i < inputs.size(); i++) {
            kernel::feature_transformer_bwd( //
                weights.get_gradients(),
                biases.get_gradients(),
                output.get_gradients(),
                inputs[i]->get_output(),
                inputs[i]->get_size(),
                i);
        }
    }

    std::vector<Ptr<Layer>> get_inputs() override {
        return {};
    }

  private:
    std::vector<Ptr<Input>> inputs;
};

} // namespace nn
