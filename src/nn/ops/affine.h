#pragma once

#include "ops.h"

namespace nn {

class Affine : public Operation {
  public:
    Affine(Ptr<Param> param, Ptr<Operation> input)
        : param(param),
          input(input) {

        name = "affine";

        input_dim = param->get_input_dim();
        output_dim = param->get_output_dim();
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

    std::vector<Ptr<Operation>> get_inputs() const override { return {input}; }

    Ptr<Param> get_param() override { return param; }

  private:
    Ptr<Param> param;
    Ptr<Operation> input;
};

} // namespace nn
