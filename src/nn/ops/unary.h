#pragma once

#include "ops.h"

namespace nn::op {

template <typename Op>
class Unary : public Operation {
  public:
    Unary(Operation* input, Op op)
        : input_(input),
          op_(op) {

        CHECK(input);

        input_dim_ = input->output_dim();
        output_dim_ = input->output_dim();
    }

    void forward() override { kernel::ElemwiseUnary<Op>::forward(input_->data(), data(), op_); }
    void backward() override { kernel::ElemwiseUnary<Op>::backward(input_->output(), grad(), op_); }

    std::vector<Operation*> inputs() const override { return {input_}; }

  private:
    Op op_;
    Operation* input_;
};

} // namespace nn::op
