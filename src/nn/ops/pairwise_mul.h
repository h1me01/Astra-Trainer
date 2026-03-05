#pragma once

#include "concat.h"
#include "ops.h"

namespace nn::op {

class PairwiseMul : public Operation {
  public:
    PairwiseMul(Operation* input)
        : input(input) {

        CHECK(input);

        name = "pairwise_mul";

        if (input->get_output_dim() % 2 != 0)
            error("PairwiseMul: Input dimension must be even!");

        input_dim = input->get_output_dim();
        output_dim = input_dim / 2;
    }

    void forward() override { kernel::pairwise_mul_fwd(input->get_data(), output.get_data(), act_type); }

    void backward() override { kernel::pairwise_mul_bwd(input->get_output(), output, act_type); }

    std::vector<Operation*> get_inputs() const override { return {input}; }

  private:
    Operation* input;
};

} // namespace nn::op