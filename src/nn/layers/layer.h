#pragma once

#include "../../nn/data.h"

#include <cstdint>
#include <string>
#include <vector>

class LayerBase {
  protected:
    std::string name;

    SparseBatch sparse_batch{1, 1};
    Tensor dense_output{1, 1};

  public:
    void init(int batch_size) {
        sparse_batch = SparseBatch(batch_size, 32);
        dense_output = Tensor(getOutputSize(), batch_size);
    }

    SparseBatch &getSparseBatch() {
        return sparse_batch;
    }

    Tensor &getDenseOutput() {
        return dense_output;
    }

    virtual ActivationType getActivationType() const = 0;

    virtual int getOutputSize() const = 0;
    virtual int getInputSize() const = 0;

    virtual std::vector<Tensor *> getTunables() = 0;

    virtual void forward() = 0;
    virtual void backprop() = 0;

    std::string getInfo() {
        std::stringstream info;
        info << name << "<";
        info << getActivationName(getActivationType()) << ">(";
        info << "input_size=" << std::to_string(getInputSize());
        info << ", output_size=" << std::to_string(getOutputSize()) << ")\n";

        const std::vector<Tensor *> &tunables = getTunables();
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
