#pragma once

#include "concat.h"
#include "ops.h"

namespace nn::op {

class PairwiseMul : public Operation {
  public:
    PairwiseMul(Operation* input)
        : input_(input) {

        CHECK(input);

        input_dim_ = input->output_dim();
        output_dim_ = input_dim_ / 2;

        if (input_dim_ % 2 != 0)
            error("PairwiseMul: Input dimension must be even!");
    }

    void forward() override { kernel::pairwise_mul_fwd(input_->data(), data()); }
    void backward() override { kernel::pairwise_mul_bwd(input_->output(), output_); }

    std::vector<Operation*> inputs() const override { return {input_}; }

  private:
    Operation* input_;
};

} // namespace nn::op