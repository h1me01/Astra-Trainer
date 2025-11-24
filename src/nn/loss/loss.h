#pragma once

#include <string>

#include "../../data/include.h"
#include "../../kernel/include.h"

namespace nn {

class Loss {
  public:
    Loss(ActivationType act_type) : act_type(act_type) {}
    virtual ~Loss() = default;

    virtual void compute(const Array<float> &target, LayerTensor &output) = 0;

    float get_loss() {
        loss.dev_to_host();
        return loss(0);
    }

    void reset() {
        loss.clear();
    }

  protected:
    Array<float> loss{1};
    ActivationType act_type;
};

} // namespace nn
