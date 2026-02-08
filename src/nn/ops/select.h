#pragma once

#include "ops.h"

namespace nn {

class Select : public Operation {
  public:
    Select(Ptr<Operation> input, Ptr<SelectIndices> indices)
        : input(input),
          indices(indices) {

        input_dim = input->get_output_dim();
        output_dim = input_dim / indices->partitions_size();

        if (input_dim % indices->partitions_size() != 0)
            error("Select input dimension must be a multiple of select size!");
    }

    void init(int batch_size) override { Operation::init(batch_size); }

    void forward() override { kernel::select_fwd(input->get_data(), output.get_data(), *indices, act_type); }

    void backward() override {
        // have to clear all input grads since only certain selected grads will be overwritten
        input->get_grads().clear_dev();
        kernel::select_bwd(input->get_grads(), output, *indices, act_type);
    }

    Ptr<SelectIndices> get_select_indices() const override { return indices; }

    std::vector<Ptr<Operation>> get_inputs() const override { return {input}; }

  private:
    int max_indices;

    Ptr<Operation> input;
    Ptr<SelectIndices> indices;
};

} // namespace nn
