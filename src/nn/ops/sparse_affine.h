#pragma once

#include "concat.h"
#include "input.h"
#include "ops.h"

namespace nn::op {

class SparseAffineBase : public Operation {
  public:
    SparseAffineBase(std::string op_name, Param* param, Input* input, int out_dim_divisor = 1)
        : param(param),
          input(input) {

        CHECK(param && input);
        name = op_name;

        input_dim = param->get_input_dim();
        output_dim = param->get_output_dim() / out_dim_divisor;

        if (input_dim % 768 != 0)
            error("SparseAffine: input dimension must be a multiple of 768!");
    }

    void init(int batch_size) override {
        if (!concat)
            Operation::init(batch_size);
    }

    void fuse_with_concat(FusedConcat* c) {
        concat = c;
        out_offset = concat->fuse(this);
    }

    Param* get_param() override { return param; }
    Input* get_input() const { return input; }

  protected:
    DenseMatrix& effective_weights() {
        return param->has_factorizer() ? param->get_factorizer().get_weights() : param->get_weights().get_data();
    }

    Tensor& effective_output() { return concat ? concat->get_output() : output; }

    int out_offset = 0;
    FusedConcat* concat = nullptr;

    Param* param;
    Input* input;
};

class SparseAffine : public SparseAffineBase {
  public:
    SparseAffine(Param* param, Input* input)
        : SparseAffineBase("sparse_affine", param, input) {}

    void forward() override {
        kernel::sparse_affine_fwd(
            effective_weights(),
            param->get_biases().get_data(),
            effective_output().get_data(),
            input->get_indices(),
            input->size(),
            out_offset,
            act_type
        );
    }

    void backward() override {
        kernel::sparse_affine_bwd(
            param->get_weights().get_grads(),
            param->get_biases().get_grads(),
            effective_output(),
            input->get_indices(),
            input->size(),
            out_offset,
            act_type
        );
    }
};

class SparseAffinePairwiseMul : public SparseAffineBase {
  public:
    SparseAffinePairwiseMul(Param* param, Input* input)
        : SparseAffineBase("sparse_affine_pairwise_mul", param, input, 2) {}

    void forward() override {
        kernel::sparse_affine_pairwise_mul_fwd(
            effective_weights(),
            param->get_biases().get_data(),
            effective_output().get_data(),
            input->get_indices(),
            input->size(),
            out_offset,
            act_type
        );
    }

    void backward() override {
        kernel::sparse_affine_pairwise_mul_bwd(
            effective_weights(),
            param->get_weights().get_grads(),
            param->get_biases(),
            effective_output().get_grads(),
            input->get_indices(),
            input->size(),
            out_offset,
            act_type
        );
    }
};

} // namespace nn::op
