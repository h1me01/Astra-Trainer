#pragma once

#include "layer.h"

namespace nn {

class Concat : public Layer {
  public:
    explicit Concat(const Ptr<Layer>& input1, const Ptr<Layer>& input2)
        : input1(input1),
          input2(input2) {}

    void init(int batch_size) override {
        input_size = input1->get_output_size() + input2->get_output_size();
        output_size = input1->get_output_size() + input2->get_output_size();
        Layer::init(batch_size);
    }

    void forward() override {
        kernel::concat_fwd(
            input1->get_output(), input2->get_output(), output.get_linear_output(), output.get_activated(), act_type
        );
    }

    void backward() override {
        kernel::concat_bwd(
            input1->get_gradients(),
            input2->get_gradients(),
            output.get_linear_output(),
            output.get_gradients(),
            act_type
        );
    }

    std::vector<Ptr<Layer>> get_inputs() override {
        return {input1, input2};
    }

  private:
    Ptr<Layer> input1;
    Ptr<Layer> input2;
};

} // namespace nn
