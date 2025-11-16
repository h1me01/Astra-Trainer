#pragma once

#include "../../kernel/activation/activation.h"

namespace nn {

class Activation {
  public:
    Activation() : act_type(ActivationType::Linear) {}

    void init(int size, int batch_size) {
        if(is_some())
            output = Tensor<float>(size, batch_size);
    }

    void forward(const DenseMatrix<float> &input) {
        if(is_some())
            kernel::activate_fwd(input, output.get_values(), act_type);
    }

    void backward(Tensor<float> &input) {
        if(is_some())
            kernel::activate_bwd(input, output.get_gradients(), act_type);
    }

    bool is_some() const {
        return act_type != ActivationType::Linear;
    }

    void set_activation_type(ActivationType type) {
        act_type = type;
    }

    Tensor<float> &get_output() {
        return output;
    }

  private:
    ActivationType act_type;
    Tensor<float> output{0, 0};
};

} // namespace nn
