#pragma once

#include "concat.h"
#include "ops.h"

namespace nn::op {

class SparseAffine : public Operation {
  public:
    SparseAffine(SPtr<Param> params, SPtr<Input> input)
        : params(params),
          input(input) {

        name = "sparse_affine";

        input_dim = params->get_input_dim();
        output_dim = params->get_output_dim();

        if (input_dim % 768 != 0)
            error("SparseAffine: input dimension must be a multiple of 768!");
    }

    void forward() override {
        auto concat_ptr = concat.lock();
        auto& real_output = concat_ptr ? concat_ptr->get_output() : output;

        if (pairwise_fused) {
            kernel::sparse_affine_pairwise_mul_fwd(
                params->get_weights().get_data(),
                params->get_biases().get_data(),
                real_output.get_data(),
                input->get_output(),
                input->get_size(),
                out_offset * output_dim / 2,
                act_type
            );
        } else {
            kernel::sparse_affine_fwd(
                params->get_weights().get_data(),
                params->get_biases().get_data(),
                real_output.get_data(),
                input->get_output(),
                input->get_size(),
                out_offset * output_dim,
                act_type
            );
        }
    }

    void backward() override {
        auto concat_ptr = concat.lock();
        auto& real_output = concat_ptr ? concat_ptr->get_output() : output;

        if (pairwise_fused) {
            kernel::sparse_affine_pairwise_mul_bwd(
                params->get_weights(),
                params->get_biases(),
                real_output,
                input->get_output(),
                input->get_size(),
                out_offset * output_dim / 2,
                act_type
            );
        } else {
            kernel::sparse_affine_bwd(
                params->get_weights().get_grads(),
                params->get_biases().get_grads(),
                real_output,
                input->get_output(),
                input->get_size(),
                out_offset * output_dim,
                act_type
            );
        }
    }

    void clear_grads() override {
        if (concat.expired())
            output.get_grads().clear_dev();
    }

    void set_concat(SPtr<Concat> concat, bool pairwise_fused = false) {
        this->concat = concat;
        this->pairwise_fused = pairwise_fused;
        out_offset = concat->fuse(shared_from_this(), pairwise_fused);
        output.free(); // not needed anymore
    }

    Tensor& get_output() override {
        if (!concat.expired())
            error("Cannot use non existing output! (This should never happen)");
        return output;
    }

    const Tensor& get_output() const override {
        if (!concat.expired())
            error("Cannot use non existing output! (This should never happen)");
        return output;
    }

    SPtr<Param> get_param() override { return params; }

    SPtr<Input> get_input() const { return input; }

  private:
    int out_offset = 0;
    bool pairwise_fused = false;

    SPtr<Param> params;
    WPtr<Concat> concat;
    SPtr<Input> input;
};

} // namespace nn::op
