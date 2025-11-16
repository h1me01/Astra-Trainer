#pragma once

#include "layer.h"

namespace nn {

class PairwiseMul : public Layer {
  public:
    explicit PairwiseMul(const Ptr<Layer> &input) : input(input) {}

    void init(int batch_size) override {
        input_size = input->get_output_size();
        if(input_size % 2 != 0)
            error("PairwiseMul layer input size must be even!");
        output_size = input->get_output_size() / 2;
        Layer::init(batch_size);
    }

    void forward() override {
        kernel::pairwise_mul_fwd( //
            input->get_output().get_values(),
            output.get_values());

        activation.forward(output.get_values());
    }

    void backward() override {
        activation.backward(output);

        kernel::pairwise_mul_bwd( //
            input->get_output(),
            output.get_gradients());
    }

    std::vector<Ptr<Layer>> get_inputs() override {
        return {input};
    }

  private:
    Ptr<Layer> input;
};

} // namespace nn
