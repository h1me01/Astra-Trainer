#pragma once

#include "layer.h"

namespace nn {

class Concat : public Layer {
  public:
    Concat(const Ptr<Layer> &input1, const Ptr<Layer> &input2) //
        : input1(input1), input2(input2) {}

    void init(int batch_size) override {
        input_size = input1->get_output_size() + input2->get_output_size();
        output_size = input1->get_output_size() + input2->get_output_size();
        Layer::init(batch_size);
    }

    void forward() override {
        kernel::concat_fwd( //
            input1->get_output().get_values(),
            input2->get_output().get_values(),
            output.get_values());

        activation.forward(output.get_values());
    }

    void backward() override {
        activation.backward(output);

        kernel::concat_bwd( //
            input1->get_output().get_gradients(),
            input2->get_output().get_gradients(),
            output.get_gradients());
    }

    std::vector<Ptr<Layer>> get_inputs() override {
        return {input1, input2};
    }

  private:
    Ptr<Layer> input1;
    Ptr<Layer> input2;
};

} // namespace nn
