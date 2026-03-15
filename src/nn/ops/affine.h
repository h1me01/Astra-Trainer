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

        input_dim_ = param->get_input_dim();
        output_dim_ = param->get_output_dim();

        if (input->get_output_dim() != input_dim_)
            error("Affine: Input output dimension does not match affine input dimension!");
    }

    void forward() override {
        kernel::affine_fwd(
            param_->get_weights().get_data(),
            param_->get_biases().get_data(),
            input_->get_data(),
            output_.get_data(),
            act_type_
        );
    }

    void backward() override {
        kernel::affine_bwd(param_->get_weights(), param_->get_biases(), input_->get_output(), output_, act_type_);
    }

    std::vector<Operation*> get_inputs() const override { return {input_}; }

    Param* get_param() override { return param_.get(); }

  private:
    SPtr<Param> param_;
    Operation* input_;
};

} // namespace nn::op
