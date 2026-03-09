#pragma once

#include "../ops/include.h"

namespace nn::loss {

class Loss {
  public:
    Loss(ActivationType act)
        : act_type(act) {}

    virtual ~Loss() = default;

    void init(int batch_size) { targets = Array<float>(batch_size, true); }

    virtual void compute(Tensor& output) = 0;

    float get() {
        loss.dev_to_host();
        return loss(0);
    }

    void clear() { loss.clear_dev(); }

    Array<float>& get_targets() { return targets; }

  protected:
    ActivationType act_type;
    Array<float> targets;
    Array<float> loss{1, true};
};

} // namespace nn::loss
