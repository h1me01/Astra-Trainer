#pragma once

#include "ops.h"

namespace nn::op {

class Select : public Operation {
  public:
    Select(SPtr<Operation> input, SPtr<SelectIndices> indices)
        : input(input),
          indices(indices) {

        name = "select";

        input_dim = input->get_output_dim();
        output_dim = input_dim / indices->partitions_size();

        if (input_dim % indices->partitions_size() != 0)
            error("Select input dimension must be a multiple of select size!");
    }

    void init(int batch_size) override { Operation::init(batch_size); }

    void forward() override { kernel::select_fwd(input->get_data(), output.get_data(), *indices, act_type); }

    void backward() override { kernel::select_bwd(input->get_grads(), output, *indices, act_type); }

    SPtr<SelectIndices> get_indices() const { return indices; }

    std::vector<SPtr<Operation>> get_inputs() const override { return {input}; }

  private:
    int max_indices;

    SPtr<Operation> input;
    SPtr<SelectIndices> indices;
};

} // namespace nn::op
