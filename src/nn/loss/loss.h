#pragma once

#include "../ops/include.h"

namespace nn {

class Loss {
  public:
    Loss(Activation act_type)
        : act_type(act_type) {}

    virtual ~Loss() = default;

    virtual void compute(const Array<float>& target, OpTensor& output) = 0;

    float get_loss() {
        loss.dev_to_host();
        return loss(0);
    }

    void reset() { loss.clear(); }

  protected:
    Activation act_type;
    Array<float> loss{1};
};

} // namespace nn
