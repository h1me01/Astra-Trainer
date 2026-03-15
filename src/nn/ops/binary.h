#pragma once

#include "ops.h"

namespace nn::op {

template <typename Op>
class Binary : public Operation {
  public:
    Binary(Operation* input1, Operation* input2, Op op)
        : input1_(input1),
          input2_(input2),
          op_(op) {

        CHECK(input1);
        CHECK(input2);

        input_dim_ = input1->output_dim();
        output_dim_ = input1->output_dim();
    }

    void forward() override { kernel::ElemwiseBinary<Op>::forward(input1_->data(), input2_->data(), data(), op_); }
    void backward() override {
        kernel::ElemwiseBinary<Op>::backward(input1_->output(), input2_->output(), grad(), op_);
    }

    std::vector<Operation*> inputs() const override { return {input1_, input2_}; }

  private:
    Op op_;
    Operation* input1_;
    Operation* input2_;
};

} // namespace nn::op
