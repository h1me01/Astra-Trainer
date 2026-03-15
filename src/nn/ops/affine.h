#pragma once

#include "ops.h"

namespace nn::op {

class Affine : public Operation {
  public:
    Affine(SPtr<Param> param, Operation* input)
        : param_(param),
          input_(input) {

        CHECK(param && input);

        name_ = "affine";

        input_dim_ = param->input_dim();
        output_dim_ = param->output_dim();

        if (input->output_dim() != input_dim_)
            error("Affine: Input output dimension does not match affine input dimension!");
    }

    void forward() override {
        kernel::affine_fwd(
            param_->weights().data(), param_->biases().data(), input_->data(), data(), act_type_
        );
    }

    void backward() override {
        kernel::affine_bwd(param_->weights(), param_->biases(), input_->output(), output_, act_type_);
    }

    std::vector<Operation*> inputs() const override { return {input_}; }

    Param* param() override { return param_.get(); }

  private:
    SPtr<Param> param_;
    Operation* input_;
};

} // namespace nn::op
