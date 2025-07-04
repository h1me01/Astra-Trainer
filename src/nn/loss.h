#pragma once

#include <string>

#include "../kernel/include.h"
#include "data/include.h"

class Loss {
  protected:
    Array<float> loss{1};

  public:
    Loss() {}

    virtual void apply(const Array<float> &target, Tensor &output) = 0;
    virtual std::string info() = 0;

    float get_loss() {
        loss.dev_to_host();
        return loss(0);
    }

    void reset() {
        loss.clear();
    }
};

template <ActivationType act_type> //
struct MPELoss : Loss {
  private:
    float m_power;

  public:
    MPELoss(float power) : Loss(), m_power(power) {}

    void apply(const Array<float> &targets, Tensor &output) {
        mpe_loss( //
            targets,
            loss,
            output,
            m_power,
            act_type,
            output.get_data().size());
    }

    std::string info() {
        return "MPELoss<" + get_activation_name(act_type) + ">(power=" + format_number(m_power) + ")";
    }
};

template <ActivationType act_type> //
struct MSELoss : Loss {
    MSELoss() : Loss() {}

    void apply(const Array<float> &targets, Tensor &output) {
        mse_loss( //
            targets,
            loss,
            output,
            act_type,
            output.get_data().size());
    }

    std::string info() {
        return "MSELoss<" + get_activation_name(act_type) + ">()";
    }
};