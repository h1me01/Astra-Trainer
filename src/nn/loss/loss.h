#pragma once

#include "../ops/include.h"

namespace nn {

class Loss {
  public:
    Loss(Activation act)
        : act_type(act) {}

    virtual ~Loss() = default;

    virtual void compute(const Array<float>& target, Tensor& output) = 0;

    float get_loss() {
        loss.dev_to_host();
        return loss(0);
    }

    void reset() { loss.clear_dev(); }

  protected:
    Activation act_type;
    Array<float> loss{1};
};

} // namespace nn
