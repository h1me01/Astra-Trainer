#pragma once

#include "ops.h"

namespace nn {

class Concat : public Operation {
  public:
    Concat(Ptr<Operation> input1, Ptr<Operation> input2)
        : input1(input1),
          input2(input2) {

        name = "concat";

        input_dim = input1->get_output_dim() + input2->get_output_dim();
        output_dim = input_dim;
    }

    void forward() override { kernel::concat_fwd(input1->get_data(), input2->get_data(), output.get_data(), act_type); }

    void backward() override { kernel::concat_bwd(input1->get_grads(), input2->get_grads(), output, act_type); }

    std::vector<Ptr<Operation>> get_inputs() const override { return {input1, input2}; }

  private:
    Ptr<Operation> input1;
    Ptr<Operation> input2;
};

} // namespace nn
