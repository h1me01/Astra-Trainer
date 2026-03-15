#pragma once

#include "ops.h"

namespace nn::op {

template <typename Op>
class Unary : public Operation {
  public:
    Unary(Operation* input)
        : input_(input) {

        CHECK(input);

        input_dim_ = input->output_dim();
        output_dim_ = input->output_dim();
    }

    void forward() override { kernel::ElemwiseUnary<Op>::forward(input_->data(), data()); }
    void backward() override { kernel::ElemwiseUnary<Op>::backward(input_->output(), grad()); }

    std::vector<Operation*> inputs() const override { return {input_}; }

  private:
    Operation* input_;
};

} // namespace nn::op
