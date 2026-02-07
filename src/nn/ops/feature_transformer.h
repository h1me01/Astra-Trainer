#pragma once

#include "ops.h"

namespace nn {

class Input {
  public:
    Input(int size)
        : size(size) {}

    void init(int batch_size) { output = Array<int>(size * batch_size, true); }

    Array<int>& get_output() { return output; }

    const Array<int>& get_output() const { return output; }

    int get_size() const { return size; }

  private:
    int size;
    Array<int> output;
};

class FeatureTransformer : public Operation {
  public:
    FeatureTransformer(Ptr<Param> params, Ptr<Input> input)
        : FeatureTransformer(params, input, nullptr) {}

    FeatureTransformer(Ptr<Param> params, Ptr<Input> input1, Ptr<Input> input2)
        : params(params) {

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
                params->get_weights().get_values(),
                params->get_biases().get_values(),
                tensor_output.get_linear_output(),
                tensor_output.get_activated(),
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
                params->get_weights().get_gradients(),
                params->get_biases().get_gradients(),
                tensor_output.get_gradients(),
                tensor_output.get_linear_output(),
                inputs[i]->get_output(),
                inputs[i]->get_size(),
                i * (output_dim / input_count),
                act_type
            );
        }
    }

    Ptr<Param> get_param() override { return params; }

  private:
    Ptr<Param> params;
    std::vector<Ptr<Input>> inputs;
};

} // namespace nn
