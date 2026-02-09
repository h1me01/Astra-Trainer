#pragma once

#include "ops.h"

namespace nn {

class FeatureTransformer : public Operation {
  public:
    FeatureTransformer(Ptr<Param> params, Ptr<Input> input)
        : FeatureTransformer(params, input, nullptr) {
            name = "feature_transformer";
        }

    // output will be concatenation of the two inputs
    FeatureTransformer(Ptr<Param> params, Ptr<Input> input1, Ptr<Input> input2)
        : params(params) {

        name = "feature_transformer_fused";

        inputs.push_back(input1);
        if (input2)
            inputs.push_back(input2);

        input_dim = params->get_input_dim();
        output_dim = inputs.size() * params->get_output_dim();

        if (input_dim % 768 != 0)
            error("FeatureTransformer input dimension must be a multiple of 768!");
    }

    void forward() override {
        const int input_count = inputs.size();
        for (int i = 0; i < input_count; i++) {
            kernel::feature_transformer_fwd(
                params->get_weights().get_data(),
                params->get_biases().get_data(),
                output.get_data(),
                inputs[i]->get_output(),
                inputs[i]->get_size(),
                i * (output_dim / input_count),
                act_type
            );
        }
    }

    void backward() override {
        const int input_count = inputs.size();
        for (int i = 0; i < input_count; i++) {
            kernel::feature_transformer_bwd(
                params->get_weights().get_grads(),
                params->get_biases().get_grads(),
                output,
                inputs[i]->get_output(),
                inputs[i]->get_size(),
                i * (output_dim / input_count),
                act_type
            );
        }
    }

    Ptr<Param> get_param() override { return params; }

    std::vector<Ptr<Input>> get_inputs_ft() const { return inputs; }

  private:
    Ptr<Param> params;
    std::vector<Ptr<Input>> inputs;
};

} // namespace nn
