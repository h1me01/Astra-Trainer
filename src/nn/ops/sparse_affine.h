#pragma once

#include "concat.h"
#include "input.h"
#include "ops.h"

namespace nn::op {

class SparseAffineBase : public Operation {
  public:
    SparseAffineBase(std::string op_name, SPtr<Param> param, Input* input, int out_dim_divisor = 1)
        : param_(param),
          input_(input) {

        CHECK(param && input);
        name_ = op_name;

        input_dim_ = param->get_input_dim();
        output_dim_ = param->get_output_dim() / out_dim_divisor;
    }

    void init(int batch_size) override {
        if (!concat_)
            Operation::init(batch_size);
    }

    void fuse_with_concat(FusedConcat* c) {
        concat_ = c;
        out_offset_ = concat_->fuse(this);
    }

    Param* get_param() override { return param_.get(); }
    Input* get_input() const { return input_; }

  protected:
    int out_offset_ = 0;
    FusedConcat* concat_ = nullptr;

    SPtr<Param> param_;
    Input* input_;

    DenseMatrix& effective_weights() {
        return param_->has_factorizer() ? param_->get_factorizer().get_weights() : param_->get_weights().get_data();
    }

    Tensor& effective_output() { return concat_ ? concat_->get_output() : output_; }
};

struct SparseAffine : public SparseAffineBase {
    SparseAffine(SPtr<Param> param, Input* input)
        : SparseAffineBase("sparse_affine", param, input) {}

    void forward() override {
        kernel::sparse_affine_fwd(
            effective_weights(),
            param_->get_biases().get_data(),
            effective_output().get_data(),
            input_->get_indices(),
            out_offset_,
            act_type_
        );
    }

    void backward() override {
        kernel::sparse_affine_bwd(
            param_->get_weights().get_grads(),
            param_->get_biases().get_grads(),
            effective_output(),
            input_->get_indices(),
            out_offset_,
            act_type_
        );
    }
};

struct SparseAffinePairwiseMul : public SparseAffineBase {
    SparseAffinePairwiseMul(SPtr<Param> param, Input* input)
        : SparseAffineBase("sparse_affine_pairwise_mul", param, input, 2) {}

    void forward() override {
        kernel::sparse_affine_pairwise_mul_fwd(
            effective_weights(),
            param_->get_biases().get_data(),
            effective_output().get_data(),
            input_->get_indices(),
            out_offset_,
            act_type_
        );
    }

    void backward() override {
        kernel::sparse_affine_pairwise_mul_bwd(
            effective_weights(),
            param_->get_weights().get_grads(),
            param_->get_biases(),
            effective_output().get_grads(),
            input_->get_indices(),
            out_offset_,
            act_type_
        );
    }
};

} // namespace nn::op
