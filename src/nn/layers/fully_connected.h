#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "../../kernel/kernel.h"
#include "../../misc.h"
#include "../data.h"
#include "layer.h"

template <int size, ActivationType act_type = Linear> //
class FullyConnected : public LayerBase {
  private:
    Tensor weights{1, 1};
    Tensor biases{size, 1};

    LayerBase *previous;

  public:
    FullyConnected(LayerBase *previous, WeightInitType init_type) : previous(previous) {
        name = "FullyConnected";

        int input_size = previous->get_output_size();

        weights = Tensor(size, input_size);
        switch(init_type) {
        case WeightInitType::Uniform:
            weights.init_uniformly();
            break;
        case WeightInitType::He:
            weights.he_init(input_size);
            break;
        }
    }

    void forward() override {
        Tensor &inputs = previous->get_output().activated;

        affine_fwd( //
            weights.get_vals(),
            biases.get_vals(),
            inputs.get_vals(),
            output.activated.get_vals(),
            output.pre_activated,
            act_type);
    }

    void backward() override {
        Tensor &inputs = previous->get_output().activated;

        affine_bwd( //
            weights,
            biases,
            inputs,
            output.activated,
            output.pre_activated,
            act_type);
    }

    ActivationType activation_type() const override {
        return act_type;
    }

    int get_output_size() const override {
        return size;
    }

    int get_input_size() const override {
        return previous->get_output_size();
    }

    std::vector<Tensor *> get_params() override {
        return {&weights, &biases};
    }

    std::string get_info() override {
        std::stringstream ss;
        ss << name << "<";
        ss << get_activation_name(activation_type()) << ">(";
        ss << std::to_string(get_input_size());
        ss << "->" << std::to_string(size) << ")\n";
        ss << params_info();
        return ss.str();
    }
};
