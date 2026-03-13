#pragma once

#include "../ops/include.h"

namespace nn::loss {

class Loss {
  public:
    Loss(ActivationType act)
        : act_type(act) {}

    virtual ~Loss() = default;

    virtual void compute(Tensor& output, const Array<float>& targets) = 0;

    float get() {
        loss.dev_to_host();
        return loss(0);
    }

    void clear() { loss.clear_dev(); }

  protected:
    ActivationType act_type;
    Array<float> loss{1, true};
};

} // namespace nn::loss
