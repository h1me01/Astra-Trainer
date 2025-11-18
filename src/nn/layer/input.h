#pragma once

#include "../../data/include.h"

namespace nn {

class Input {
  public:
    explicit Input(int size) : size(size) {}

    void init(int batch_size) {
        output = Array<int>(size * batch_size);
    }

    Array<int> &get_output() {
        return output;
    }

    int get_size() const {
        return size;
    }

  private:
    int size;
    Array<int> output;
};

} // namespace nn
