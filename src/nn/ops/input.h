#pragma once

#include "ops.h"

namespace nn::op {

class Input : public Operation {
  public:
    Input(int size) {
        if (size <= 0)
            error("Input: Size must be positive!");
        output_dim = size;
        name = "Input";
    }

    void init(int batch_size) override { sparse_indices = Array<int>(output_dim * batch_size, true); }

    void forward() override {}
    void backward() override {}

    Array<int>& get_indices() { return sparse_indices; }
    const Array<int>& get_indices() const { return sparse_indices; }

    int size() const { return output_dim; }

  private:
    Array<int> sparse_indices;
};

} // namespace nn::op
