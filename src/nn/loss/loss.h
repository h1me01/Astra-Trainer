#pragma once

#include "../ops/include.h"

namespace nn::loss {

class Loss {
  public:
    Loss(ActivationType act)
        : act_type_(act) {}

    virtual ~Loss() = default;

    virtual void compute(Tensor& output, const Array<float>& targets) = 0;

    float get() {
        loss_.dev_to_host();
        return loss_(0);
    }

    void clear() { loss_.clear_dev(); }

  protected:
    ActivationType act_type_;
    Array<float> loss_{1, true};
};

} // namespace nn::loss
