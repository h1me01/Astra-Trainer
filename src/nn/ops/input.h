#pragma once

#include "ops.h"

namespace nn::op {

class Input : public Operation {
  public:
    Input(int size) {
        if (size <= 0)
            error("Input: Size must be positive!");
        output_dim_ = size;
        name_ = "Input";
    }

    void init(int batch_size) override {
        indices_ = SparseMatrix(output_dim_, batch_size);
        indices_.pinned();
    }

    void forward() override {}
    void backward() override {}

    SparseMatrix& get_indices() { return indices_; }
    const SparseMatrix& get_indices() const { return indices_; }

    int& operator()(int r, int c) { return indices_(r, c); }

    void reset() {
        for (int i = 0; i < indices_.size(); i++)
            indices_(i) = -1;
    }

  private:
    SparseMatrix indices_;
};

} // namespace nn::op
