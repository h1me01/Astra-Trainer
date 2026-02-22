#pragma once

#include "ops.h"

namespace nn::op {

class Concat : public Operation {
  public:
    Concat(std::vector<SPtr<Operation>> inputs)
        : inputs(inputs) {

        if (inputs.size() < 2)
            error("Concat: requires at least 2 inputs!");

        name = "concat";

        for (const auto& input : inputs)
            input_dim += input->get_output_dim();

        output_dim = input_dim;
    }

    void forward() override {
        if (skip)
            return;
        int offset = 0;
        for (const auto& input : inputs) {
            kernel::concat_fwd(input->get_data(), output.get_data(), offset, act_type);
            offset += input->get_output_dim();
        }
    }

    void backward() override {
        if (skip)
            return;
        int offset = 0;
        for (const auto& input : inputs) {
            kernel::concat_bwd(input->get_grads(), output, offset, act_type);
            offset += input->get_output_dim();
        }
    }

    std::vector<SPtr<Operation>> get_inputs() const override { return inputs; }

    int fuse(SPtr<Operation> op) {
        CHECK(should_skip());
        CHECK(
            get_activation() == Activation::Linear        //
            || op->get_activation() == Activation::Linear //
        );

        int offset = 0;
        for (int i = 0; i < get_inputs().size(); i++) {
            if (get_inputs()[i].get() == op.get())
                return offset;
            offset += get_inputs()[i]->get_output_dim();
        }

        error("Concat fusion failed! (this should never happen)");

        return -1;
    }

    void set_skip() { skip = true; }
    bool should_skip() const { return skip; }

  private:
    bool skip = false;
    std::vector<SPtr<Operation>> inputs;
};

} // namespace nn::op
