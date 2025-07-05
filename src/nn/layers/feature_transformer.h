#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "../../kernel/include.h"
#include "../../misc.h"
#include "layer.h"

template <int size, ActivationType act_type> //
class FeatureTransformer : public LayerBase {
  private:
    int input_size;
    Tensor weights{1, 1};
    Tensor biases{size, 1};

  public:
    FeatureTransformer(int input_size, WeightInitType init_type) : input_size(input_size) {
        if(input_size % 768 != 0)
            error("Error: input size must be divisible by 768 to match standard chess inputs.");

        name = "FeatureTransformer";

        weights = Tensor(size, input_size);
        weights.init(init_type, input_size);
    }

    void forward() override {
        const std::vector<Array<int>> &features = sparse_batch.get_features();

        int i = 0;
        for(auto &feature : features) {
            sparse_affine_fwd( //
                output.activated.get_data(),
                output.pre_activated,
                weights.get_data(),
                biases.get_data(),
                feature,
                sparse_batch.get_feature_sizes(),
                i * size,
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
        int bucket_size = input_size / 768;

        std::stringstream ss;
        ss << name << "<";
        ss << get_activation_name(activation_type()) << ">([";
        ss << std::to_string(bucket_size) << "x768";
        ss << "->" << std::to_string(size) << "]x2):\n";
        ss << params_info();
        return ss.str();
    }
};
