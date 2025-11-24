#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../../data/include.h"
#include "../../kernel/include.h"
#include "../../misc.h"
#include "../../training_data_format/include.h"

namespace nn {

class Layer : public std::enable_shared_from_this<Layer> {
  public:
    Layer() = default;
    virtual ~Layer() = default;

    Layer(int input_size, int output_size, WeightInitType winit_type)
        : input_size(input_size), output_size(output_size) {
        is_main = true;
        weights = Tensor(output_size, input_size);
        biases = Tensor(output_size, 1);
        weights.init(winit_type, input_size);
    }

    virtual void init(int batch_size) {
        ASSERT(output_size > 0 && input_size > 0);
        output.init(output_size, batch_size, has_activation(act_type));
    }

    virtual void step(const std::vector<TrainingDataEntry> &data_entries) {
        output.clear_grads();
    }

    virtual void forward() = 0;
    virtual void backward() = 0;

    void clamp_weights(float min, float max) {
        weights.clamp(min, max);
    }

    void clamp_biases(float min, float max) {
        biases.clamp(min, max);
    }

    Ptr<Layer> relu() {
        act_type = ActivationType::ReLU;
        return shared_from_this();
    }

    Ptr<Layer> crelu() {
        act_type = ActivationType::CReLU;
        return shared_from_this();
    }

    Ptr<Layer> screlu() {
        act_type = ActivationType::SCReLU;
        return shared_from_this();
    }

    Ptr<Layer> sigmoid() {
        act_type = ActivationType::Sigmoid;
        return shared_from_this();
    }

    int get_output_size() const {
        return output_size;
    }

    int get_input_size() const {
        return input_size;
    }

    Tensor &get_weights() {
        return is_main ? weights : main->get_weights();
    }

    Tensor &get_biases() {
        return is_main ? biases : main->get_biases();
    }

    DenseMatrix &get_output() {
        return output.get_output();
    }

    DenseMatrix &get_gradients() {
        return output.get_gradients();
    }

    LayerTensor &get_layer_tensor() {
        return output;
    }

    std::vector<Tensor *> get_params() {
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
    ActivationType act_type = ActivationType::Linear;

    Tensor weights;
    Tensor biases;
    LayerTensor output;

    // main layers are created by the user (not including helper layers)
    bool is_main = false;
    Ptr<Layer> main;
};

} // namespace nn
