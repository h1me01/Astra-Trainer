#pragma once

#include "ops.h"

namespace nn {

class Activate : public Operation {
  public:
    Activate(Ptr<Operation> input, Activation type)
        : input(input),
          type(type) {

        name = "activate";

        input_dim = input->get_output_dim();
        output_dim = input->get_output_dim();
    }

    void forward() override { kernel::activation_fwd(input->get_data(), output.get_data(), type); }

    void backward() override { kernel::activation_bwd(input->get_output(), output.get_grads(), type); }

    std::vector<Ptr<Operation>> get_inputs() const override { return {input}; }

  private:
    Activation type;
    Ptr<Param> param;
    Ptr<Operation> input;
};

} // namespace nn
