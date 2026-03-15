#pragma once

#include "ops.h"

namespace nn::op {

class Select : public Operation {
  public:
    Select(Operation* input, SPtr<SelectIndices> indices)
        : input_(input),
          indices_(indices) {

        CHECK(input && indices);

        name_ = "select";

        input_dim_ = input->output_dim();
        output_dim_ = input_dim_ / indices->partitions_size();

        if (input_dim_ % indices->partitions_size() != 0)
            error("Select: input dimension must be a multiple of select size!");
    }

    void init(int batch_size) override { Operation::init(batch_size); }

    void forward() override { kernel::select_fwd(input_->data(), output_.data(), *indices_, act_type_); }
    void backward() override { kernel::select_bwd(input_->grad(), output_, *indices_, act_type_); }

    SelectIndices* indices() const { return indices_.get(); }

    std::vector<Operation*> inputs() const override { return {input_}; }

  private:
    int max_indices_;

    Operation* input_;
    SPtr<SelectIndices> indices_;
};

} // namespace nn::op
