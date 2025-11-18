#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "input.h"
#include "layer.h"

namespace nn {

class FeatureTransformer : public Layer {
  public:
    explicit FeatureTransformer(int input_size, int output_size, WeightInitType winit_type = WeightInitType::He)
        : Layer(input_size, output_size, winit_type) {

        if(input_size % 768 != 0)
            error("Input size must be divisible by 768 to match standard chess inputs!");
    }

    explicit FeatureTransformer(FeatureTransformer &other, const Ptr<Input> &input) {
        this->main = other.get_main();
        this->inputs = {input};
        this->input_size = other.input_size;
        this->output_size = other.output_size;
        ASSERT(main != nullptr);
    }

    explicit FeatureTransformer(FeatureTransformer &other, const Ptr<Input> &input1, const Ptr<Input> &input2)
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

        for(int i = 0; i < inputs.size(); i++) {
            kernel::feature_transformer_fwd( //
                get_weights().get_values(),
                get_biases().get_values(),
                output.get_values(),
                inputs[i]->get_output(),
                inputs[i]->get_size(),
                i * (output_size / 2));
        }

        activation.forward(output.get_values());
    }

    void backward() override {
        if(is_main)
            return; // main layer does not perform backward

        activation.backward(output);

        for(int i = 0; i < inputs.size(); i++) {
            kernel::feature_transformer_bwd( //
                get_weights().get_gradients(),
                get_biases().get_gradients(),
                output.get_gradients(),
                inputs[i]->get_output(),
                inputs[i]->get_size(),
                i * (output_size / 2));
        }
    }

    std::vector<Ptr<Layer>> get_inputs() override {
        return {};
    }

  private:
    std::vector<Ptr<Input>> inputs;
};

} // namespace nn
