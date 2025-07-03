#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "../../kernel/kernel.h"
#include "../../misc.h"
#include "../data.h"
#include "layer.h"

template <int size, ActivationType act_type> //
class DualFeatureTransformer : public LayerBase {
  private:
    int input_size;
    Tensor weights{1, 1};
    Tensor biases{size, 1};

  public:
    DualFeatureTransformer(int input_size, WeightInitType init_type) : input_size(input_size) {
        name = "FeatureTransformer";

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
        const std::vector<Array<int>> &features = sparse_batch.get_features();

        int i = 0;
        for(auto &feature : features) {
            sparse_affine_fwd( //
                output.activated.get_vals(),
                output.pre_activated,
                weights.get_vals(),
                biases.get_vals(),
                feature,
                sparse_batch.get_feature_sizes(),
                i * size,
                sparse_batch.get_batch_size(),
                sparse_batch.get_max_entries(),
                act_type);

            i++;
        }
    }

    void backward() override {
        const std::vector<Array<int>> &features = sparse_batch.get_features();

        int i = 0;
        for(auto &feature : features) {
            sparse_affine_bwd( //
                output.activated.get_grads(),
                output.pre_activated,
                weights.get_grads(),
                biases.get_grads(),
                feature,
                sparse_batch.get_feature_sizes(),
                i * size,
                sparse_batch.get_batch_size(),
                sparse_batch.get_max_entries(),
                act_type);

            i++;
        }
    }

    ActivationType activation_type() const override {
        return act_type;
    }

    int get_output_size() const override {
        return 2 * size;
    }

    int get_input_size() const override {
        return input_size;
    }

    std::vector<Tensor *> get_params() override {
        return {&weights, &biases};
    }

    std::string get_info() override {
        std::stringstream ss;
        ss << name << "<";
        ss << get_activation_name(activation_type()) << ">(";
        ss << std::to_string(get_input_size());
        ss << "->2x" << std::to_string(size) << ")\n";
        ss << params_info();
        return ss.str();
    }
};
