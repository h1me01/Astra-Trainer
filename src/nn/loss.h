#pragma once

#include <string>

#include "../kernel/kernel.h"
#include "data.h"

class Loss {
  protected:
    Array<float> m_loss;

  public:
    Loss() : m_loss(1) {}

    virtual void apply(const Array<float> &target, Tensor &output) = 0;
    virtual std::string info() = 0;

    float loss() {
        m_loss.dev_to_host();
        return m_loss(0);
    }

    void reset() {
        m_loss(0) = 0;
        m_loss.host_to_dev();
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
            m_loss,
            output,
            m_power,
            act_type,
            output.get_vals().size());
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
            m_loss,
            output,
            act_type,
            output.get_vals().size());
    }

    std::string info() {
        return "MSELoss<" + get_activation_name(act_type) + ">()";
    }
};