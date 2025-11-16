#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../../data/include.h"
#include "../../kernel/include.h"
#include "../../misc.h"
#include "../batch_data/batch_data.h"
#include "../utils/utils.h"
#include "activation.h"

namespace nn {

class Layer : public std::enable_shared_from_this<Layer> {
  public:
    Layer() = default;
    virtual ~Layer() = default;

    Layer(int input_size, int output_size, WeightInitType init_type)
        : input_size(input_size), output_size(output_size) {
        is_main = true;
        weights = Tensor<float>(output_size, input_size);
        biases = Tensor<float>(output_size, 1);
        weights.init(init_type, input_size);
    }

    virtual void init(int batch_size) {
        ASSERT(output_size > 0 && input_size > 0);
        output = Tensor<float>(output_size, batch_size);
        activation.init(output_size, batch_size);
    }

    virtual void step() {}

    virtual void forward() = 0;
    virtual void backward() = 0;

    void clamp_weights(float min, float max) {
        weights.clamp(min, max);
    }

    void clamp_biases(float min, float max) {
        biases.clamp(min, max);
    }

    Ptr<Layer> relu() {
        activation.set_activation_type(ActivationType::ReLU);
        return shared_from_this();
    }

    Ptr<Layer> crelu() {
        activation.set_activation_type(ActivationType::CReLU);
        return shared_from_this();
    }

    Ptr<Layer> screlu() {
        activation.set_activation_type(ActivationType::SCReLU);
        return shared_from_this();
    }

    Ptr<Layer> sigmoid() {
        activation.set_activation_type(ActivationType::Sigmoid);
        return shared_from_this();
    }

    int get_output_size() const {
        return output_size;
    }

    int get_input_size() const {
        return input_size;
    }

    Tensor<float> &get_output() {
        return activation.is_some() ? activation.get_output() : output;
    }

    std::vector<Tensor<float> *> get_params() {
        if(is_main)
            return {&weights, &biases};
        else
            return {};
    }

    Ptr<Layer> get_main() {
        if(is_main)
            return shared_from_this();
        else
            return main;
    }

    virtual std::vector<Ptr<Layer>> get_inputs() = 0;

  protected:
    int input_size = 0;
    int output_size = 0;
    Activation activation;

    Tensor<float> weights{0, 0};
    Tensor<float> biases{0, 0};
    Tensor<float> output{0, 0};

    // main layers are created by the user (not including helper layers)
    bool is_main = false;
    Ptr<Layer> main;

    void set_main(const Ptr<Layer> &main_layer) {
        this->main = main_layer;
    }
};

} // namespace nn
