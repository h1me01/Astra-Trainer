#pragma once

#include "../../kernel/include.h"

namespace nn::param {

class Factorizer {
  public:
    Factorizer(Tensor* param_weights, int block_size)
        : param_weights(param_weights) {

        int rows = param_weights->rows();
        int cols = param_weights->cols();

        if (cols % block_size != 0)
            error("Factorizer: block_size must divide the number of columns in the weight matrix.!");

        base = Tensor(rows, block_size);
        weights = DenseMatrix(rows, cols);
    }

    void forward() { kernel::factorizer_fwd(base.get_data(), param_weights->get_data(), weights); }
    void backward() { kernel::factorizer_bwd(base.get_grads(), param_weights->get_grads()); }

    Tensor& get_base() { return base; }
    DenseMatrix& get_weights() { return weights; }

  private:
    Tensor* param_weights;
    Tensor base;
    DenseMatrix weights;
};

} // namespace nn::param
