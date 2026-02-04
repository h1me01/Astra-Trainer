#pragma once

#include "ops.h"

namespace nn {

class PairwiseMul : public Operation {
  public:
    PairwiseMul(Ptr<Operation> input)
        : PairwiseMul(input, nullptr) {}

    PairwiseMul(Ptr<Operation> input1, Ptr<Operation> input2) {
        inputs.push_back(input1);
        if (input2)
            inputs.push_back(input2);

        input_dim = input1->get_output_dim();
        output_dim = (input_dim / 2) * inputs.size();

        if (input_dim % 2 != 0)
            error("PairwiseMul input dimension must be even!");
        if (inputs.size() == 2 && input1->get_output_dim() != input2->get_output_dim())
            error("PairwiseMul input dimensions must match!");
    }

    void forward() override {
        for (int i = 0; i < (int)inputs.size(); i++) {
            kernel::pairwise_mul_fwd(
                inputs[i]->get_output(),
                tensor_output.get_linear_output(),
                tensor_output.get_activated(),
                i * (input_dim / 2),
                act_type
            );
        }
    }

    void backward() override {
        for (int i = 0; i < (int)inputs.size(); i++) {
            kernel::pairwise_mul_bwd(
                inputs[i]->get_output(),
                inputs[i]->get_gradients(),
                tensor_output.get_linear_output(),
                tensor_output.get_gradients(),
                i * (input_dim / 2),
                act_type
            );
        }
    }

    std::vector<Ptr<Operation>> get_inputs() override {
        return inputs;
    }

  private:
    std::vector<Ptr<Operation>> inputs;
};

} // namespace nn