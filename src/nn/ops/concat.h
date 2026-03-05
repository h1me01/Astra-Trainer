#pragma once

#include "ops.h"

namespace nn::op {

class ConcatBase : public Operation {
  public:
    ConcatBase(std::string op_name, std::vector<Operation*>& inputs)
        : inputs(inputs) {
        if (inputs.size() < 2)
            error("Concat: requires at least 2 inputs!");
        for (const auto& input : inputs)
            CHECK(input);
        name = std::move(op_name);
        for (const auto& input : inputs)
            output_dim += input->get_output_dim();
        input_dim = output_dim;
    }

    std::vector<Operation*> get_inputs() const override { return inputs; }

  protected:
    std::vector<Operation*> inputs;
};

struct Concat : public ConcatBase {
    Concat(std::vector<Operation*>& inputs)
        : ConcatBase("concat", inputs) {}

    void forward() override {
        int offset = 0;
        for (const auto& input : inputs) {
            kernel::concat_fwd(input->get_data(), output.get_data(), offset, act_type);
            offset += input->get_output_dim();
        }
    }

    void backward() override {
        int offset = 0;
        for (const auto& input : inputs) {
            kernel::concat_bwd(input->get_grads(), output, offset, act_type);
            offset += input->get_output_dim();
        }
    }
};

struct FusedConcat : public ConcatBase {
    FusedConcat(std::vector<Operation*>& inputs)
        : ConcatBase("fused_concat", inputs) {}

    void forward() override {}
    void backward() override {}

    int fuse(Operation* op) {
        int offset = 0;
        for (const auto& input : inputs) {
            if (input == op)
                return offset;
            offset += input->get_output_dim();
        }
        error("Concat fusion failed! (this should never happen)");
        return -1;
    }
};

} // namespace nn::op
