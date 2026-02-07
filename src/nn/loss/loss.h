#pragma once

#include "../ops/include.h"

namespace nn {

class Loss : public std::enable_shared_from_this<Loss> {
  public:
    virtual ~Loss() = default;

    virtual void compute(const Array<float>& target, OpTensor& output) = 0;

    float get_loss() {
        loss.dev_to_host();
        return loss(0);
    }

    void reset() { loss.clear(); }

    Ptr<Loss> relu() {
        act_type = Activation::ReLU;
        return shared_from_this();
    }

    Ptr<Loss> crelu() {
        act_type = Activation::CReLU;
        return shared_from_this();
    }

    Ptr<Loss> screlu() {
        act_type = Activation::SCReLU;
        return shared_from_this();
    }

    Ptr<Loss> sigmoid() {
        act_type = Activation::Sigmoid;
        return shared_from_this();
    }

  protected:
    Activation act_type = Activation::Linear;
    Array<float> loss{1};
};

} // namespace nn
