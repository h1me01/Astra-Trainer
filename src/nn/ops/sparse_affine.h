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

    void init(int batch_size) override {
        if (concat.expired())
            Operation::init(batch_size);
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
                out_offset,
                act_type
            );
        } else {
            kernel::sparse_affine_fwd(
                params->get_weights().get_data(),
                params->get_biases().get_data(),
                real_output.get_data(),
                input->get_output(),
                input->get_size(),
                out_offset,
                act_type
            );
        }
    }

    void backward() override {
        auto concat_ptr = concat.lock();
        auto& real_output = concat_ptr ? concat_ptr->get_output() : output;

        if (pairwise_fused) {
            kernel::sparse_affine_pairwise_mul_bwd(
                params->get_weights().get_data(),
                params->get_weights().get_grads(),
                params->get_biases(),
                real_output,
                input->get_output(),
                input->get_size(),
                out_offset,
                act_type
            );
        } else {
            kernel::sparse_affine_bwd(
                params->get_weights().get_grads(),
                params->get_biases().get_grads(),
                real_output,
                input->get_output(),
                input->get_size(),
                out_offset,
                act_type
            );
        }
    }

    void set_concat(SPtr<Concat> concat) {
        this->concat = concat;
        out_offset = concat->fuse(shared_from_this());
    }

    void set_pairwise_fused() {
        output_dim /= 2;
        pairwise_fused = true;
    }

    bool is_pairwise_fused() const { return pairwise_fused; }

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
