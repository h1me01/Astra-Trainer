#pragma once

#include "ops.h"

namespace nn::op {

class Affine : public Operation {
  public:
    Affine(SPtr<Param> param, Operation* input)
        : param(param),
          input(input) {

        CHECK(param && input);

        name = "affine";

        input_dim = param->get_input_dim();
        output_dim = param->get_output_dim();

        if (input->get_output_dim() != input_dim)
            error("Affine: input output dimension does not match affine input dimension!");
    }

    void forward() override {
        kernel::affine_fwd(
            param->get_weights().get_data(),
            param->get_biases().get_data(),
            input->get_data(),
            output.get_data(),
            act_type
        );
    }

    void backward() override {
        kernel::affine_bwd(param->get_weights(), param->get_biases(), input->get_output(), output, act_type);
    }

    std::vector<Operation*> get_inputs() const override { return {input}; }

    Param* get_param() override { return param.get(); }

  private:
    SPtr<Param> param;
    Operation* input;
};

} // namespace nn::op
