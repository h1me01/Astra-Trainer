#pragma once

#include "ops.h"

namespace nn::op {

class ConcatBase : public Operation {
  public:
    ConcatBase(std::vector<Operation*>& inputs)
        : inputs_(inputs) {

        if (inputs.size() < 2)
            error("Concat: requires at least 2 inputs!");
        for (const auto& input : inputs)
            CHECK(input);
        for (const auto& input : inputs)
            output_dim_ += input->output_dim();
        input_dim_ = output_dim_;
    }

    std::vector<Operation*> inputs() const override { return inputs_; }

  protected:
    std::vector<Operation*> inputs_;
};

struct Concat : public ConcatBase {
    Concat(std::vector<Operation*>& inputs)
        : ConcatBase(inputs) {}

    void forward() override {
        int offset = 0;
        for (const auto& input : inputs_) {
            kernel::concat_fwd(input->data(), data(), offset);
            offset += input->output_dim();
        }
    }

    void backward() override {
        int offset = 0;
        for (const auto& input : inputs_) {
            kernel::concat_bwd(input->grad(), output_, offset);
            offset += input->output_dim();
        }
    }
};

struct FusedConcat : public ConcatBase {
    FusedConcat(std::vector<Operation*>& inputs)
        : ConcatBase(inputs) {}

    void forward() override {}
    void backward() override {}

    int fuse(Operation* op) {
        int offset = 0;
        for (const auto& input : inputs_) {
            if (input == op)
                return offset;
            offset += input->output_dim();
        }
        CHECK(false);
        return -1;
    }
};

} // namespace nn::op
