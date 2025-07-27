#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "../../kernel/include.h"
#include "../../misc.h"
#include "layer.h"

template <int size, ActivationType act_type = Linear> //
class Affine : public LayerBase {
  public:
    Affine(LayerBase *previous, WeightInitType init_type) : previous(previous) {
        name = "Affine";

        const int input_size = previous->get_output_size();

        weights = Tensor<float>(size, input_size);
        weights.init(init_type, input_size);
    }

    void forward() override {
        auto &inputs = previous->get_output().activated;

        affine_fwd( //
            weights.get_data(),
            biases.get_data(),
            inputs.get_data(),
            output.activated.get_data(),
            output.pre_activated,
            act_type);
    }

    void backward() override {
        auto &inputs = previous->get_output().activated;

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

    std::vector<Tensor<float> *> get_params() override {
        return {&weights, &biases};
    }

    std::string get_info() override {
        std::stringstream ss;
        ss << name << "<";
        ss << get_activation_name(activation_type()) << ">(";
        ss << std::to_string(get_input_size());
        ss << "->" << std::to_string(size) << "):\n";
        ss << params_info();
        return ss.str();
    }

  private:
    Tensor<float> weights{1, 1};
    Tensor<float> biases{size, 1};

    LayerBase *previous;
};
