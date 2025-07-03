#pragma once

#include "../../nn/data.h"

#include <cstdint>
#include <string>
#include <vector>

enum class WeightInitType { Uniform, He };

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

    std::string params_info() {
        std::stringstream info;

        const std::vector<Tensor *> &params = this->get_params();
        if(!params.empty()) {
            int i = 0;
            for(const Tensor *t : params) {
                std::string prefix = (i++ == 0) ? "weights(" : "biases(";
                info << "  -> " << prefix                   //
                     << "min=" << format_number(t->min())   //
                     << ", max=" << format_number(t->max()) //
                     << ")" << "\n";
            }
        }

        return info.str();
    }

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

    virtual std::vector<Tensor *> get_params() = 0;

    virtual void forward() = 0;
    virtual void backward() = 0;

    virtual std::string get_info() = 0;
};
