#pragma once

#include "ops.h"

namespace nn::op {

class Activate : public Operation {
  public:
    Activate(Operation* input, ActivationType type)
        : input(input),
          type(type) {

        CHECK(input);

        name = "activate_";

        switch (type) {
        case ActivationType::ReLU:
            name += "relu";
            break;
        case ActivationType::ClippedReLU:
            name += "clipped_relu";
            break;
        case ActivationType::SqrClippedReLU:
            name += "sqr_clipped_relu";
            break;
        case ActivationType::Sigmoid:
            name += "sigmoid";
            break;
        default:
            name += "linear";
            break;
        }

        input_dim = input->get_output_dim();
        output_dim = input->get_output_dim();
    }

    void forward() override { kernel::activation_fwd(input->get_data(), output.get_data(), type); }

    void backward() override { kernel::activation_bwd(input->get_output(), output.get_grads(), type); }

    std::vector<Operation*> get_inputs() const override { return {input}; }

  private:
    ActivationType type;
    Operation* input;
};

} // namespace nn::op
