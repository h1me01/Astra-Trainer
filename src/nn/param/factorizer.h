#pragma once

#include "../../kernel/include.h"

namespace nn::param {

class Factorizer {
  public:
    Factorizer(Tensor* param_weights)
        : param_weights(param_weights),
          base(param_weights->get_data().rows(), 768),
          weights(param_weights->get_data().rows(), param_weights->get_data().cols()) {}

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
