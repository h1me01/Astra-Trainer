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

    void init(int batch_size) override {
        indices = SparseMatrix(output_dim, batch_size);
        indices.pinned();
    }

    void forward() override {}
    void backward() override {}

    SparseMatrix& get_indices() { return indices; }
    const SparseMatrix& get_indices() const { return indices; }

    int& operator()(int r, int c) { return indices(r, c); }

    void reset() {
        for (int i = 0; i < indices.size(); i++)
            indices(i) = -1;
    }

  private:
    SparseMatrix indices;
};

} // namespace nn::op
