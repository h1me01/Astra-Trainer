#pragma once

#include "../graph/common.h"
#include "../ops/include.h"
#include "../util.h"

namespace nn::loss {

class Loss {
  public:
    Loss(graph::OpType act_type) { act_op_ = nn::util::get_activation_op(act_type); }

    virtual ~Loss() = default;

    virtual void compute(Tensor& output, const Array<float>& targets) = 0;

    float get() {
        loss_.dev_to_host();
        return loss_(0);
    }

    void clear() { loss_.clear_dev(); }

  protected:
    kernel::ActOp act_op_;
    Array<float> loss_{1, true};
};

} // namespace nn::loss
