#pragma once

#include "concat.h"
#include "ops.h"

namespace nn::op {

class PairwiseMul : public Operation {
  public:
    PairwiseMul(Operation* input)
        : input_(input) {

        CHECK(input);

        name_ = "pairwise_mul";

        input_dim_ = input->output_dim();
        output_dim_ = input_dim_ / 2;

        if (input_dim_ % 2 != 0)
            error("PairwiseMul: Input dimension must be even!");
    }

    void forward() override { kernel::pairwise_mul_fwd(input_->data(), output_.data(), act_type_); }
    void backward() override { kernel::pairwise_mul_bwd(input_->output(), output_, act_type_); }

    std::vector<Operation*> inputs() const override { return {input_}; }

  private:
    Operation* input_;
};

} // namespace nn::op