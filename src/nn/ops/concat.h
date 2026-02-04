#pragma once

#include "ops.h"

namespace nn {

class Concat : public Operation {
  public:
    Concat(Ptr<Operation> input1, Ptr<Operation> input2)
        : input1(input1),
          input2(input2) {

        input_dim = input1->get_output_dim() + input2->get_output_dim();
        output_dim = input_dim;
    }

    void forward() override {
        kernel::concat_fwd(
            input1->get_output(),
            input2->get_output(),
            tensor_output.get_linear_output(),
            tensor_output.get_activated(),
            act_type
        );
    }

    void backward() override {
        kernel::concat_bwd(
            input1->get_gradients(),
            input2->get_gradients(),
            tensor_output.get_linear_output(),
            tensor_output.get_gradients(),
            act_type
        );
    }

    std::vector<Ptr<Operation>> get_inputs() override {
        return {input1, input2};
    }

  private:
    Ptr<Operation> input1;
    Ptr<Operation> input2;
};

} // namespace nn
