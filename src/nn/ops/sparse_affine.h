#pragma once

#include "concat.h"
#include "ops.h"

namespace nn::op {

class SparseAffine : public Operation {
  public:
    SparseAffine(SPtr<Param> params, SPtr<Input> input)
        : param(params),
          input(input) {

        name = "sparse_affine";

        input_dim = params->get_input_dim();
        output_dim = params->get_output_dim();

        if (has_factorizer())
            factorized_output = DenseMatrix(output_dim, input_dim);

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

        if (has_factorizer()) {
            int offset = 0;
            for (int i = 0; i < input_dim / 768; i++) {
                kernel::factorizer_fwd(
                    get_factorizer_weights().get_data(), param->get_weights().get_data(), factorized_output, offset
                );
                offset += 768 * param->get_output_dim();
            }
        }

        auto& real_weights = has_factorizer() ? factorized_output : param->get_weights().get_data();
        if (pairwise_fused) {
            kernel::sparse_affine_pairwise_mul_fwd(
                real_weights,
                param->get_biases().get_data(),
                real_output.get_data(),
                input->get_output(),
                input->get_size(),
                out_offset,
                act_type
            );
        } else {
            kernel::sparse_affine_fwd(
                real_weights,
                param->get_biases().get_data(),
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
            auto& real_weights = has_factorizer() ? factorized_output : param->get_weights().get_data();

            kernel::sparse_affine_pairwise_mul_bwd(
                real_weights,
                param->get_weights().get_grads(),
                param->get_biases(),
                real_output,
                input->get_output(),
                input->get_size(),
                out_offset,
                act_type
            );
        } else {
            kernel::sparse_affine_bwd(
                param->get_weights().get_grads(),
                param->get_biases().get_grads(),
                real_output,
                input->get_output(),
                input->get_size(),
                out_offset,
                act_type
            );
        }

        if (has_factorizer()) {
            int offset = 0;
            for (int i = 0; i < input_dim / 768; i++) {
                kernel::factorizer_bwd(get_factorizer_weights().get_grads(), param->get_weights().get_grads(), offset);
                offset += 768 * param->get_output_dim();
            }
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

    SPtr<Param> get_param() override { return param; }

    SPtr<Input> get_input() const { return input; }

  private:
    int out_offset = 0;
    bool pairwise_fused = false;

    SPtr<Param> param;
    WPtr<Concat> concat;
    SPtr<Input> input;
    DenseMatrix factorized_output;

    Tensor& get_factorizer_weights() { return param->get_factorizer_weights(); }
    bool has_factorizer() const { return param->get_factorizer_weights().size() > 0; }
};

} // namespace nn::op
