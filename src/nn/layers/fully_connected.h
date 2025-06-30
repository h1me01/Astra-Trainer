#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "../../kernel/kernel.h"
#include "../../misc.h"
#include "../data.h"
#include "layer.h"

template < //
    int size,
    ActivationType act_type,
    bool bucketed = false>
class FullyConnected : public LayerBase {
  private:
    static constexpr int bucket_size = 8; // for now only supports 8 buckets

    Tensor weights{1, 1};
    Tensor biases{size * (bucketed ? bucket_size : 1), 1};

    LayerBase *previous;

  public:
    FullyConnected(LayerBase *previous, WeightInitType init_type) : previous(previous) {
        name = "FullyConnected";

        int input_size = previous->getOutputSize();

        weights = Tensor(size * (bucketed ? bucket_size : 1), input_size);
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
        Tensor &inputs = previous->getDenseOutput().activated;

        if(bucketed) {
            bucketed_affine_fwd( //
                weights.getValues(),
                biases.getValues(),
                inputs.getValues(),
                output.activated.getValues(),
                output.pre_activated,
                sparse_batch.getPSQTIndices(),
                act_type,
                bucket_size);
        } else {
            affine( //
                weights.getValues(),
                biases.getValues(),
                inputs.getValues(),
                output.activated.getValues(),
                output.pre_activated,
                act_type);
        }
    }

    void backprop() override {
        Tensor &inputs = previous->getDenseOutput().activated;

        if(bucketed) {
            bucketed_affine_bwd( //
                weights,
                biases,
                inputs,
                output.activated,
                output.pre_activated,
                sparse_batch.getPSQTIndices(),
                act_type,
                bucket_size);
        } else {
            affine_bp( //
                weights,
                biases,
                inputs,
                output.activated,
                output.pre_activated,
                act_type);
        }
    }

    ActivationType getActivationType() const override {
        return act_type;
    }

    int getOutputSize() const override {
        return size;
    }

    int getInputSize() const override {
        return previous->getOutputSize();
    }

    std::vector<Tensor *> getParams() override {
        return {&weights, &biases};
    }

    std::string getInfo() override {
        std::stringstream info;
        info << name << "<";
        info << getActivationName(getActivationType()) << ">(";
        info << std::to_string(getInputSize());
        info << "->" << (bucketed ? "8x" : "") << std::to_string(size) << ")\n";
        info << getParamsInfo();
        return info.str();
    }
};
