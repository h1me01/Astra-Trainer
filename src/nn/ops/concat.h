#pragma once

#include "ops.h"

namespace nn {

class Concat : public Operation {
  public:
    Concat(std::vector<SPtr<Operation>> inputs)
        : inputs(inputs) {

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

    int fuse(SPtr<Operation> op, bool sparse_affine_pairwise_mul_fusion = false) {
        ASSERT(should_skip());
        ASSERT(
            get_activation() == Activation::Linear        //
            || op->get_activation() == Activation::Linear //
        );

        for (int i = 0; i < get_inputs().size(); i++) {
            if (sparse_affine_pairwise_mul_fusion) {
                if (get_inputs()[i]->get_inputs()[0].get() == op.get())
                    return i;
            } else {
                if (get_inputs()[i].get() == op.get())
                    return i;
            }
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

} // namespace nn
