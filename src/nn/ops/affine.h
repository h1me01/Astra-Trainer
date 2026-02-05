#pragma once

#include "ops.h"

namespace nn {

class Affine : public Operation {
  public:
    Affine(Ptr<Params> params, Ptr<Operation> input)
        : params(params),
          input(input) {

        input_dim = params->get_input_dim();
        output_dim = params->get_output_dim();
    }

    void forward() override {
        kernel::affine_fwd(
            params->get_weights().get_values(),
            params->get_biases().get_values(),
            input->get_output(),
            tensor_output.get_linear_output(),
            tensor_output.get_activated(),
            act_type
        );
    }

    void backward() override {
        kernel::affine_bwd(
            params->get_weights(),
            params->get_biases(),
            input->get_output(),
            input->get_gradients(),
            tensor_output.get_linear_output(),
            tensor_output.get_gradients(),
            act_type
        );
    }

    Ptr<Params> get_params() override { return params; }

  private:
    Ptr<Params> params;
    Ptr<Operation> input;
};

} // namespace nn
