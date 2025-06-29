#pragma once

#include "../../nn/data.h"

#include <cstdint>
#include <string>
#include <vector>

class LayerBase {
  protected:
    std::string name;

    static SparseBatch sparse_batch;

    struct Output {
        DenseMatrix pre_activated{1, 1};
        Tensor activated{1, 1};

        Output() {}
        Output(int output_size, int batch_size)
            : pre_activated(output_size, batch_size), activated(output_size, batch_size) {}
    };

    Output output;

  public:
    void init(int batch_size) {
        sparse_batch = SparseBatch(batch_size, 32);
        output = Output(getOutputSize(), batch_size);
    }

    SparseBatch &getSparseBatch() {
        return sparse_batch;
    }

    Output &getDenseOutput() {
        return output;
    }

    void clampWeights(float min, float max) {
        getParams()[0]->clamp(min, max);
    }

    void clampBiases(float min, float max) {
        getParams()[1]->clamp(min, max);
    }

    virtual ActivationType getActivationType() const = 0;

    virtual int getOutputSize() const = 0;
    virtual int getInputSize() const = 0;

    virtual std::vector<Tensor *> getParams() = 0;

    virtual void forward() = 0;
    virtual void backprop() = 0;

    std::string getInfo() {
        std::stringstream info;
        info << name << "<";
        info << getActivationName(getActivationType()) << ">(";
        info << "input_size=" << std::to_string(getInputSize());
        info << ", output_size=" << std::to_string(getOutputSize()) << ")\n";

        const std::vector<Tensor *> &tunables = getParams();
        if(!tunables.empty()) {
            int i = 0;
            for(const Tensor *t : tunables) {
                std::string prefix = (i++ == 0) ? "weights(" : "biases(";
                // clang-format off
                info << "  -> " 
                     << prefix 
                     << "min=" << formatNumber(t->min()) 
                     << ", max=" << formatNumber(t->max())
                     << ")" << "\n";
                // clang-format on
            }
        }

        return info.str();
    }
};
