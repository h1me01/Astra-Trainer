#pragma once

#include "../../data/include.h"

namespace nn {

class Input {
  public:
    Input(int size) : size(size) {}

    void init(int batch_size) {
        output = Tensor<int>(size, batch_size);
    }

    Tensor<int> &get_output() {
        return output;
    }

    int get_size() const {
        return size;
    }

  private:
    int size;
    Tensor<int> output{0, 0};
};

} // namespace nn
