#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "../../kernel/kernel.h"
#include "../../misc.h"
#include "../data.h"
#include "layer.h"

// perspective transformer layer
// meaning there are two sparse inputs for both sides
template <int size, ActivationType act_type> //
class FeatureTransformer : public LayerBase {
  private:
    int input_size;
    Tensor weights{1, 1};
    Tensor biases{size, 1};

  public:
    FeatureTransformer(int input_size, WeightInitType init_type) : input_size(input_size) {
        name = "FeatureTransformer";

        weights = Tensor(size, input_size);
        switch(init_type) {
        case WeightInitType::Uniform:
            weights.initUniformly();
            break;
        case WeightInitType::He:
            weights.heInit(input_size);
            break;
        }
    }

    void forward() override {
        const int batch_size = sparse_batch.getBatchSize();
        const int max_entries = sparse_batch.maxEntries();

        const Array<int> &feature_sizes = sparse_batch.getFeatureSizes();
        const std::vector<Array<int>> &features = sparse_batch.getFeatures();

        DenseMatrix &weights_v = weights.getValues();
        DenseMatrix &biases_v = biases.getValues();

        DenseMatrix &activated_v = output.activated.getValues();
        DenseMatrix &pre_activated = output.pre_activated;

        ASSERT(batch_size == activated_v.numCols());

        ASSERT(weights_v.devAddress() &&     //
               biases_v.devAddress() &&      //
               activated_v.devAddress() &&   //
               pre_activated.devAddress() && //
               feature_sizes.devAddress());

        const int block_size = 128;
        const int grid_size = std::ceil(float(weights_v.numRows() * batch_size) / block_size);

        int i = 0;
        for(auto &feature : features) {
            ASSERT(feature.devAddress())

            sparse_affine_kernel<<<grid_size, block_size>>>( //
                weights_v.devAddress(),
                biases_v.devAddress(),
                activated_v.devAddress(),
                pre_activated.devAddress(),
                feature.devAddress(),
                feature_sizes.devAddress(),
                weights_v.numRows(),
                activated_v.numRows(),
                i * size,
                batch_size,
                max_entries,
                act_type);

            i++;
        }
    }

    void backprop() override {
        const int batch_size = sparse_batch.getBatchSize();
        const int max_entries = sparse_batch.maxEntries();

        const Array<int> &feature_sizes = sparse_batch.getFeatureSizes();
        const std::vector<Array<int>> &features = sparse_batch.getFeatures();

        DenseMatrix &weights_g = weights.getGradients();
        DenseMatrix &biases_g = biases.getGradients();

        DenseMatrix &activated_g = output.activated.getGradients();
        DenseMatrix &pre_activated = output.pre_activated;

        ASSERT(activated_g.numCols() == batch_size);

        ASSERT(weights_g.devAddress() &&     //
               biases_g.devAddress() &&      //
               activated_g.devAddress() &&   //
               pre_activated.devAddress() && //
               feature_sizes.devAddress());

        const int block_size = 128;
        const int grid_size = std::ceil(float(weights_g.numRows() * batch_size) / block_size);

        int i = 0;
        for(auto &feature : features) {
            ASSERT(feature.devAddress());

            sparse_affine_bp_kernel<<<grid_size, block_size>>>( //
                activated_g.devAddress(),
                pre_activated.devAddress(),
                weights_g.devAddress(),
                biases_g.devAddress(),
                feature.devAddress(),
                feature_sizes.devAddress(),
                weights_g.numRows(),
                activated_g.numRows(),
                i * size,
                batch_size,
                max_entries,
                act_type);

            i++;
        }
    }

    ActivationType getActivationType() const override {
        return act_type;
    }

    int getOutputSize() const override {
        return 2 * size;
    }

    int getInputSize() const override {
        return input_size;
    }

    std::vector<Tensor *> getParams() override {
        return {&weights, &biases};
    }

    std::string getInfo() override {
        std::stringstream info;
        info << name << "<";
        info << getActivationName(getActivationType()) << ">(";
        info << std::to_string(getInputSize());
        info << "->2x" << std::to_string(size) << ")\n";
        info << getParamsInfo();
        return info.str();
    }
};
