#pragma once

#include "../../nn/data.h"

#include <cstdint>
#include <string>
#include <vector>

class LayerBase {
  protected:
    SparseBatch sparse_batch{1, 1};
    Tensor dense_output{1, 1};

  public:
    void init(int batch_size) {
        sparse_batch = SparseBatch(batch_size, 32);
        dense_output = Tensor(getOutputSize(), batch_size);
    }

    SparseBatch &getSparseBatch() {
        return sparse_batch;
    }

    Tensor &getDenseOutput() {
        return dense_output;
    }

    virtual int getOutputSize() const = 0;
    virtual int getInputSize() const = 0;

    virtual std::vector<Tensor *> getTunables() = 0;

    virtual void forward() = 0;
    virtual void backprop() = 0;

    virtual std::string getInfo() = 0;
};
