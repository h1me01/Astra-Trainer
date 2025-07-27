#pragma once

#include "../../nn/data/include.h"

#include <cstdint>
#include <string>
#include <vector>

class LayerBase {
  protected:
    static SparseBatch sparse_batch; // all layers must share the same sparse batch

    struct Output {
        DenseMatrix<float> pre_activated{1, 1};
        Tensor<float> activated{1, 1};

        Output() {}
        Output(int output_size, int batch_size)
            : pre_activated(output_size, batch_size), activated(output_size, batch_size) {}
    };

  public:
    void init(int batch_size) {
        sparse_batch = SparseBatch(batch_size, 32);
        output = Output(get_output_size(), batch_size);
    }

    SparseBatch &get_sparse_batch() {
        return sparse_batch;
    }

    Output &get_output() {
        return output;
    }

    void clamp_weights(float min, float max) {
        get_params()[0]->clamp(min, max);
    }

    void clamp_biases(float min, float max) {
        get_params()[1]->clamp(min, max);
    }

    virtual ActivationType activation_type() const = 0;

    virtual int get_output_size() const = 0;
    virtual int get_input_size() const = 0;

    virtual std::vector<Tensor<float> *> get_params() = 0;

    virtual void forward() = 0;
    virtual void backward() = 0;

    virtual std::string get_info() = 0;

  protected:
    std::string name;
    Output output;

    std::string params_info() {
        std::stringstream info;

        const auto &params = this->get_params();
        if(!params.empty()) {
            int i = 0;
            for(const auto *t : params) {
                std::string prefix = (std::string((i++ == 0) ? "Weights:" : "Biases :") + " clipped to [");
                info << "    - " << prefix                      //
                     << format_number(t->lower_bound())         //
                     << ", " << format_number(t->upper_bound()) //
                     << "]" << "\n";
            }
        }

        return info.str();
    }
};
