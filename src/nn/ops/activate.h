#pragma once

#include "ops.h"

namespace nn::op {

class Activate : public Operation {
  public:
    Activate(SPtr<Operation> input, Activation type)
        : input(input),
          type(type) {

        name = "activate_";

        switch (type) {
        case Activation::ReLU:
            name += "relu";
            break;
        case Activation::ClippedReLU:
            name += "clipped_relu";
            break;
        case Activation::SqrClippedReLU:
            name += "sqr_clipped_relu";
            break;
        case Activation::Sigmoid:
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

    std::vector<SPtr<Operation>> get_inputs() const override { return {input}; }

  private:
    Activation type;
    SPtr<Operation> input;
};

} // namespace nn::op
