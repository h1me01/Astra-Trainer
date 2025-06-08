#pragma once

#include <string>

#include "../kernel/kernel.h"
#include "data.h"

class Loss {
  protected:
    Array<float> loss;

  public:
    float scalar = 1;

    Loss() : loss(1) {}

    virtual void apply(const DenseMatrix &target, Tensor &output) = 0;
    virtual std::string getInfo() = 0;

    float getLoss() {
        loss.devToHost();
        return loss(0);
    }

    void reset() {
        loss(0) = 0;
        loss.hostToDev();
    }
};

struct MPELoss : Loss {
  private:
    float power;

  public:
    MPELoss(float power) : Loss(), power(power) {}

    void apply(const DenseMatrix &targets, Tensor &output) {
        const DenseMatrix &output_v = output.getValues();
        DenseMatrix &output_g = output.getGradients();

        ASSERT(output_v.devAddress() && //
               output_g.devAddress() && //
               targets.devAddress() &&  //
               loss.devAddress());

        constexpr int block_size = 1024;
        dim3 grid(std::ceil((float) output_v.size() / block_size));

        // clang-format off
        mpe_kernel<<<grid, block_size>>>
        (
            targets.devAddress(), 
            output_v.devAddress(), 
            output_g.devAddress(), 
            loss.devAddress(), 
            power, 
            output_v.size()
        );
        // clang-format on
    }

    std::string getInfo() {
        return "MPELoss(power=" + formatNumber(power) + ")";
    }
};

struct MSELoss : Loss {
    MSELoss() : Loss() {}

    void apply(const DenseMatrix &targets, Tensor &output) {
        const DenseMatrix &output_v = output.getValues();
        DenseMatrix &output_g = output.getGradients();

        ASSERT(output_v.devAddress() && //
               output_g.devAddress() && //
               targets.devAddress() &&  //
               loss.devAddress());

        constexpr int block_size = 1024;
        dim3 grid(std::ceil((float) output_v.size() / block_size));

        // clang-format off
        mse_kernel<<<grid, block_size>>>
        (
            targets.devAddress(), 
            output_v.devAddress(), 
            output_g.devAddress(), 
            loss.devAddress(), 
            output_v.size()
        );
        // clang-format on
    }

    std::string getInfo() {
        return "MSELoss()";
    }
};