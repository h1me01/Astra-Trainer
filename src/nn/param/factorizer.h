#pragma once

#include "../../kernel/include.h"

namespace nn::param {

class Factorizer {
  public:
    Factorizer(Tensor* param_weights, int block_size)
        : param_weights_(param_weights) {

        int rows = param_weights->rows();
        int cols = param_weights->cols();

        if (cols % block_size != 0)
            error("Factorizer: block_size must divide the number of columns in the weight matrix.!");

        base_ = Tensor(rows, block_size);
        weights_ = DenseMatrix(rows, cols);
    }

    void forward() { kernel::factorizer_fwd(base_.get_data(), param_weights_->get_data(), weights_); }
    void backward() { kernel::factorizer_bwd(base_.get_grads(), param_weights_->get_grads()); }

    Tensor& get_base() { return base_; }
    DenseMatrix& get_weights() { return weights_; }

  private:
    Tensor* param_weights_;
    Tensor base_;
    DenseMatrix weights_;
};

} // namespace nn::param
