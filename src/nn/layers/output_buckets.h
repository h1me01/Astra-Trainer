#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "../../kernel/include.h"
#include "../../misc.h"
#include "layer.h"

class OutputBuckets : public LayerBase {
  public:
    static constexpr int NUM_BUCKETS = 8; // for now only supports 8 buckets

  private:
    int size;
    LayerBase *previous;

  public:
    OutputBuckets(LayerBase *previous) : previous(previous) {
        name = "OutputBuckets";
        size = previous->get_output_size() / NUM_BUCKETS;
        if(size != 1) {
            error("OutputBuckets layer needs its previous layer to have exactly " + //
                  std::to_string(NUM_BUCKETS) + " outputs.");
        }
    }

    void forward() override {
        Tensor &input = previous->get_output().activated;

        const int batch_size = sparse_batch.get_batch_size();

        const Array<int> &bucket_indices = sparse_batch.get_psqt_indices();
        DenseMatrix<float> &input_v = input.get_data();

        ASSERT(input_v.rows() == NUM_BUCKETS);
        ASSERT(input_v.cols() == batch_size);
        ASSERT(batch_size == bucket_indices.size());

        select_fwd(input_v, output.activated.get_data(), bucket_indices);
    }

    void backward() override {
        Tensor &input = previous->get_output().activated;

        const int batch_size = sparse_batch.get_batch_size();

        const Array<int> &bucket_indices = sparse_batch.get_psqt_indices();
        DenseMatrix<float> &input_g = input.get_grads();

        ASSERT(input_g.rows() == NUM_BUCKETS);
        ASSERT(input_g.cols() == batch_size);
        ASSERT(batch_size == bucket_indices.size());

        select_bwd(input_g, output.activated.get_grads(), bucket_indices);
    }

    ActivationType activation_type() const override {
        return Linear; // doesn't use activation
    }

    int get_output_size() const override {
        return size;
    }

    int get_input_size() const override {
        return previous->get_output_size();
    }

    std::vector<Tensor *> get_params() override {
        return {};
    }

    std::string get_info() override {
        std::stringstream ss;
        ss << name << "(";
        ss << std::to_string(get_input_size());
        ss << "->" << std::to_string(size) << ")\n";
        return ss.str();
    }
};
