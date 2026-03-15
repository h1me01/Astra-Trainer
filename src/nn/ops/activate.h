#pragma once

#include "ops.h"

namespace nn::op {

class Activation : public Operation {
  public:
    Activation(Operation* input, ActivationType type)
        : input_(input),
          type_(type) {

        CHECK(input);

        switch (type) {
        case ActivationType::ReLU:
            name_ = "relu";
            break;
        case ActivationType::ClippedReLU:
            name_ = "clipped_relu";
            break;
        case ActivationType::SqrClippedReLU:
            name_ = "sqr_clipped_relu";
            break;
        case ActivationType::Sigmoid:
            name_ = "sigmoid";
            break;
        default:
            CHECK(false);
            break;
        }

        input_dim_ = input->output_dim();
        output_dim_ = input->output_dim();
    }

    void forward() override { kernel::activation_fwd(input_->data(), data(), type_); }
    void backward() override { kernel::activation_bwd(input_->output(), grad(), type_); }

    std::vector<Operation*> inputs() const override { return {input_}; }

  private:
    ActivationType type_;
    Operation* input_;
};

} // namespace nn::op
