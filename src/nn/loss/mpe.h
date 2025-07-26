#pragma once

#include "../../kernel/include.h"
#include "loss.h"

template <ActivationType act_type> //
struct MPELoss : Loss {
  private:
    float m_power;

  public:
    MPELoss(float power) : Loss(), m_power(power) {}

    void compute(const Array<float> &targets, Tensor<float> &output) {
        mpe_loss(targets, loss, output, m_power, act_type);
    }

    std::string get_info() {
        return "MPELoss<" + get_activation_name(act_type) + ">(power=" + format_number(m_power) + ")";
    }
};
