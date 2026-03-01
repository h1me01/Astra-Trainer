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

    void init(int batch_size) override {
        if (!concat)
            Operation::init(batch_size);
    }

    void forward() override {
        auto& real_output = concat ? concat->get_output() : output;
        kernel::pairwise_mul_fwd(input->get_data(), real_output.get_data(), out_offset, act_type);
    }

    void backward() override {
        auto& real_output = concat ? concat->get_output() : output;
        kernel::pairwise_mul_bwd(input->get_output(), real_output, out_offset, act_type);
    }

    void set_concat(Concat* c) {
        CHECK(c);
        concat = c;
        out_offset = concat->fuse(this);
    }

    std::vector<Operation*> get_inputs() const override { return {input}; }

  private:
    int out_offset = 0;
    Concat* concat = nullptr;
    Operation* input;
};

} // namespace nn::op