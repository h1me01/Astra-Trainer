#pragma once

#include "ops.h"

namespace nn::op {

template <typename Op>
class Binary : public Operation {
  public:
    Binary(Operation* input1, Operation* input2)
        : input1_(input1),
          input2_(input2) {

        CHECK(input1);
        CHECK(input2);

        input_dim_ = input1->output_dim();
        output_dim_ = input1->output_dim();
    }

    void forward() override { kernel::ElemwiseBinary<Op>::forward(input1_->data(), input2_->data(), data()); }
    void backward() override { kernel::ElemwiseBinary<Op>::backward(input1_->output(), input2_->output(), grad()); }

    std::vector<Operation*> inputs() const override { return {input1_, input2_}; }

  private:
    Operation* input1_;
    Operation* input2_;
};

} // namespace nn::op
