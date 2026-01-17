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
        if (input_size % 768 != 0)
            error("Input size must be divisible by 768!");
    }

    explicit FeatureTransformer(FeatureTransformer& other, const Ptr<Input>& input) {
        if (!input)
            error("Feature Transformer: Input is null!");

        this->main = other.get_main();
        this->inputs = {input};
        this->input_size = other.input_size;
        this->output_size = other.output_size;
        ASSERT(main != nullptr);
    }

    explicit FeatureTransformer(FeatureTransformer& other, const Ptr<Input>& input1, const Ptr<Input>& input2)
        : FeatureTransformer(other, input1) {
        if (!input2)
            error("Feature Transformer: Input2 is null!");

        this->inputs.push_back(input2);
        this->output_size *= 2;
    }

    Ptr<FeatureTransformer> forward(const Ptr<Input>& input) {
        if (!is_main)
            error("Forward can only be used by user defined layers!");
        return std::make_shared<FeatureTransformer>(*this, input);
    }

    Ptr<FeatureTransformer> forward(const Ptr<Input>& input1, const Ptr<Input>& input2) {
        if (!is_main)
            error("Forward can only be used by user defined layers!");
        return std::make_shared<FeatureTransformer>(*this, input1, input2);
    }

    void forward() override {
        ASSERT(!is_main);

        const int input_count = inputs.size();
        for (int i = 0; i < input_count; i++) {
            kernel::feature_transformer_fwd(
                get_weights().get_values(),
                get_biases().get_values(),
                output.get_linear_output(),
                output.get_activated(),
                inputs[i]->get_output(),
                inputs[i]->get_size(),
                i * (output_size / input_count),
                act_type
            );
        }
    }

    void backward() override {
        ASSERT(!is_main);

        const int input_count = inputs.size();
        for (int i = 0; i < input_count; i++) {
            kernel::feature_transformer_bwd(
                get_weights().get_gradients(),
                get_biases().get_gradients(),
                output.get_gradients(),
                output.get_linear_output(),
                inputs[i]->get_output(),
                inputs[i]->get_size(),
                i * (output_size / input_count),
                act_type
            );
        }
    }

    std::vector<Ptr<Layer>> get_inputs() override {
        return {};
    }

  private:
    std::vector<Ptr<Input>> inputs;
};

} // namespace nn
