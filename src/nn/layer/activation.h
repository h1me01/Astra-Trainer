#pragma once

#include "../../kernel/activation/activation.h"

namespace nn {

class Activation {
  public:
    explicit Activation() : act_type(ActivationType::Linear) {}

    void init(int size, int batch_size) {
        if(is_some())
            output = Tensor(size, batch_size);
    }

    void forward(const DenseMatrix &input) {
        if(is_some())
            kernel::activate_fwd(input, output.get_values(), act_type);
    }

    void backward(Tensor &input) {
        if(is_some())
            kernel::activate_bwd(input, output.get_gradients(), act_type);
    }

    bool is_some() const {
        return act_type != ActivationType::Linear;
    }

    void set_activation_type(ActivationType type) {
        act_type = type;
    }

    Tensor &get_output() {
        return output;
    }

  private:
    ActivationType act_type;
    Tensor output;
};

} // namespace nn
