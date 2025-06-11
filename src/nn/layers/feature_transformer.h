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
    Tensor weights{size, 1};
    Tensor biases{size, 1};

  public:
    FeatureTransformer(int input_size, bool init_uniformly = true) : input_size(input_size) {
        name = "FeatureTransformer";

        weights = Tensor(size, input_size);
        if(init_uniformly)
            weights.initUniformly();
        else
            weights.heInit(input_size);
    }

    void forward() override {
        const int batch_size = sparse_batch.getBatchSize();
        const int max_entries = sparse_batch.maxEntries();
        const Array<int> &feature_sizes = sparse_batch.getFeatureSizes();
        const std::vector<Array<int>> &features = sparse_batch.getFeatures();

        const DenseMatrix &weights_v = weights.getValues();
        const DenseMatrix &biases_v = biases.getValues();
        DenseMatrix &output_v = dense_output.getValues();

        ASSERT(batch_size == output_v.numCols());

        ASSERT(weights_v.devAddress() && //
               biases_v.devAddress() &&  //
               output_v.devAddress() &&  //
               feature_sizes.devAddress());

        constexpr int block_size = 128;
        dim3 grid(std::ceil(float(weights_v.numRows() * batch_size) / block_size));

        int i = 0;
        for(auto &feature : features) {
            ASSERT(feature.devAddress())

            // clang-format off
            sparse_affine_kernel<<<grid, block_size>>>
            (
                weights_v.devAddress(), 
                biases_v.devAddress(),
                output_v.devAddress(), 
                feature.devAddress(),
                feature_sizes.devAddress(),
                weights_v.numRows(), 
                output_v.numRows(), 
                i * size, 
                batch_size,
                max_entries, 
                act_type
            );
            // clang-format on

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

        const DenseMatrix &output_v = dense_output.getValues();
        const DenseMatrix &output_g = dense_output.getGradients();

        ASSERT(output_g.numCols() == batch_size);

        ASSERT(weights_g.devAddress() && //
               biases_g.devAddress() &&  //
               output_g.devAddress() &&  //
               output_v.devAddress() &&  //
               feature_sizes.devAddress());

        constexpr int block_size = 128;
        dim3 grid(std::ceil(float(weights_g.numRows() * batch_size) / block_size));

        int i = 0;
        for(auto &feature : features) {
            ASSERT(feature.devAddress());

            // clang-format off
            sparse_affine_bp_kernel<<<grid, block_size>>>
            (
                output_v.devAddress(), 
                output_g.devAddress(),
                weights_g.devAddress(), 
                biases_g.devAddress(),
                feature.devAddress(), 
                feature_sizes.devAddress(),
                weights_g.numRows(), 
                output_g.numRows(),
                i * size,
                batch_size, 
                max_entries, 
                act_type
            );
            // clang-format on

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

    std::vector<Tensor *> getTunables() override {
        return {&weights, &biases};
    }
};
