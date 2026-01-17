#pragma once

#include "dense_matrix.h"

namespace data {

class LayerTensor {
  public:
    void init(int output_size, int batch_size, bool has_activation) {
        linear_output = DenseMatrix(output_size, batch_size);
        gradients = DenseMatrix(output_size, batch_size);
        if (has_activation)
            activated = DenseMatrix(output_size, batch_size);
    }

    void clear_grads() {
        gradients.clear_dev();
    }

    bool has_activation() const {
        return activated.size() > 0;
    }

    DenseMatrix& get_output() {
        return has_activation() ? activated : linear_output;
    }

    DenseMatrix& get_linear_output() {
        return linear_output;
    }

    DenseMatrix& get_gradients() {
        return gradients;
    }

    DenseMatrix& get_activated() {
        return activated;
    }

  private:
    DenseMatrix linear_output;
    DenseMatrix gradients;
    DenseMatrix activated;
};

} // namespace data
