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
            error("Input size must be divisible by 768!");
    }

    explicit FeatureTransformer(FeatureTransformer &other, const Ptr<Input> &input)
        : FeatureTransformer(other, input, nullptr) {}

    explicit FeatureTransformer(FeatureTransformer &other, const Ptr<Input> &input1, const Ptr<Input> &input2) {
        this->main = other.get_main();
        this->inputs = {input1};
        if(input2)
            this->inputs.push_back(input2);

        this->input_size = other.input_size;
        this->output_size = other.output_size * (input2 ? 2 : 1);
        ASSERT(main != nullptr);
    }

    Ptr<FeatureTransformer> forward(const Ptr<Input> &input1, const Ptr<Input> &input2 = nullptr) {
        if(!input1)
            error("Feature Transformer: Input1 is null!");
        if(!is_main)
            error("Forward can only be used by user defined layers!");
        return std::make_shared<FeatureTransformer>(*this, input1, input2);
    }

    void forward() override {
        if(is_main)
            return;

        DenseMatrix *act_ptr = activation.is_some() ? &activation.get_output().get_values() : nullptr;

        for(int i = 0; i < inputs.size(); i++) {
            kernel::feature_transformer_fwd( //
                get_weights().get_values(),
                get_biases().get_values(),
                output.get_values(),
                act_ptr,
                inputs[i]->get_output(),
                inputs[i]->get_size(),
                i * (output_size / inputs.size()),
                activation.get_type());
        }
    }

    void backward() override {
        if(is_main)
            return;

        const bool use_act = activation.is_some();
        const DenseMatrix &incoming_grad = use_act ? activation.get_output().get_gradients() : output.get_gradients();
        const DenseMatrix *linear_out_ptr = use_act ? &output.get_values() : nullptr;

        for(int i = 0; i < inputs.size(); i++) {
            kernel::feature_transformer_bwd( //
                get_weights().get_gradients(),
                get_biases().get_gradients(),
                incoming_grad,
                linear_out_ptr,
                inputs[i]->get_output(),
                inputs[i]->get_size(),
                i * (output_size / inputs.size()),
                activation.get_type());
        }
    }

    std::vector<Ptr<Layer>> get_inputs() override {
        return {};
    }

  private:
    std::vector<Ptr<Input>> inputs;
};

} // namespace nn
