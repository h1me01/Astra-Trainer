#pragma once

#include "concat.h"
#include "ops.h"

namespace nn {

class PairwiseMul : public Operation {
  public:
    PairwiseMul(SPtr<Operation> input)
        : input(input) {
        name = "pairwise_mul";

        if (input->get_output_dim() % 2 != 0)
            error("Input dimension for pairwise multiplication must be even!");

        input_dim = input->get_output_dim();
        output_dim = input_dim / 2;
    }

    void forward() override {
        if (skip)
            return;
        auto concat_ptr = concat.lock();
        auto& real_output = concat_ptr ? concat_ptr->get_output() : output;
        kernel::pairwise_mul_fwd(input->get_data(), real_output.get_data(), out_offset * output_dim, act_type);
    }

    void backward() override {
        if (skip)
            return;
        auto concat_ptr = concat.lock();
        auto& real_output = concat_ptr ? concat_ptr->get_output() : output;
        kernel::pairwise_mul_bwd(input->get_output(), real_output, out_offset * output_dim, act_type);
    }

    void clear_grads() override {
        if (!skip && concat.expired())
            output.get_grads().clear_dev();
    }

    void set_concat(SPtr<Concat> concat) {
        ASSERT(!skip && concat);
        this->concat = concat;
        out_offset = concat->fuse(shared_from_this());
        output.free(); // not needed anymore
    }

    Tensor& get_output() override {
        if (!concat.expired() || skip)
            error("Cannot use non existing output! (This should never happen)");
        return output;
    }

    const Tensor& get_output() const override {
        if (!concat.expired() || skip)
            error("Cannot use non existing output! (This should never happen)");
        return output;
    }

    void set_skip() {
        skip = true;
        // we free pairwise output but not concat output
        // since pairwise will only get skipped for that
        // multi-layer fusion where only the concat output is needed
        output.free(); // not needed anymore
    }

    bool should_skip() const { return skip; }

    std::vector<SPtr<Operation>> get_inputs() const override { return {input}; }

  private:
    bool skip = false;
    int out_offset = 0;
    WPtr<Concat> concat;
    SPtr<Operation> input;
};

} // namespace nn