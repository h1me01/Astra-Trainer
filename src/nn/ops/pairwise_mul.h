#pragma once

#include "concat.h"
#include "ops.h"

namespace nn::op {

class PairwiseMul : public Operation {
  public:
    PairwiseMul(SPtr<Operation> input)
        : input(input) {
        name = "pairwise_mul";

        if (input->get_output_dim() % 2 != 0)
            error("PairwiseMul: Input dimension must be even!");

        input_dim = input->get_output_dim();
        output_dim = input_dim / 2;
    }

    void init(int batch_size) override {
        if (concat.expired())
            Operation::init(batch_size);
    }

    void forward() override {
        auto concat_ptr = concat.lock();
        auto& real_output = concat_ptr ? concat_ptr->get_output() : output;
        kernel::pairwise_mul_fwd(input->get_data(), real_output.get_data(), out_offset, act_type);
    }

    void backward() override {
        auto concat_ptr = concat.lock();
        auto& real_output = concat_ptr ? concat_ptr->get_output() : output;
        kernel::pairwise_mul_bwd(input->get_output(), real_output, out_offset, act_type);
    }

    void set_concat(SPtr<Concat> concat) {
        ASSERT(concat);
        this->concat = concat;
        out_offset = concat->fuse(shared_from_this());
    }

    std::vector<SPtr<Operation>> get_inputs() const override { return {input}; }

  private:
    int out_offset = 0;
    WPtr<Concat> concat;
    SPtr<Operation> input;
};

} // namespace nn::op