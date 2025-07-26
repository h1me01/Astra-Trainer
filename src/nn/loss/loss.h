#pragma once

#include <string>

#include "data/include.h"

class Loss {
  protected:
    Array<float> loss{1};

  public:
    Loss() {}

    virtual void compute(const Array<float> &target, Tensor<float> &output) = 0;
    virtual std::string get_info() = 0;

    float get_loss() {
        loss.dev_to_host();
        return loss(0);
    }

    void reset() {
        loss.clear();
    }
};
