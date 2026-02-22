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

        if (param->has_factorizer())
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

        if (param->has_factorizer()) {
            kernel::factorizer_fwd(
                param->get_factorizer().get_data(), param->get_weights().get_data(), factorized_output
            );
        }

        auto& real_weights = param->has_factorizer() ? factorized_output : param->get_weights().get_data();
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

    void backward() override {
        auto concat_ptr = concat.lock();
        auto& real_output = concat_ptr ? concat_ptr->get_output() : output;

        kernel::sparse_affine_bwd(
            param->get_weights().get_grads(),
            param->get_biases().get_grads(),
            real_output,
            input->get_output(),
            input->get_size(),
            out_offset,
            act_type
        );

        if (param->has_factorizer()) {
            kernel::factorizer_bwd(param->get_factorizer().get_grads(), param->get_weights().get_grads());
        }
    }

    void set_concat(SPtr<Concat> concat) {
        this->concat = concat;
        out_offset = concat->fuse(shared_from_this());
    }

    SPtr<Param> get_param() override { return param; }

    SPtr<Input> get_input() const { return input; }

  private:
    int out_offset = 0;

    SPtr<Param> param;
    WPtr<Concat> concat;
    SPtr<Input> input;
    DenseMatrix factorized_output;
};

} // namespace nn::op
