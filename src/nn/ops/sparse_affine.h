#pragma once

#include "../graph/common.h"

#include "../util.h"
#include "concat.h"
#include "input.h"
#include "ops.h"

namespace nn::op {

class SparseAffineBase : public Operation {
  public:
    SparseAffineBase(SPtr<Param> param, Input* input, int out_dim_divisor = 1)
        : param_(param),
          input_(input) {

        CHECK(param && input);

        input_dim_ = param->input_dim();
        output_dim_ = param->output_dim() / out_dim_divisor;
    }

    void init(int batch_size) override {
        if (!concat_)
            Operation::init(batch_size);
    }

    void fuse_with_concat(FusedConcat* c) {
        concat_ = c;
        out_offset_ = concat_->fuse(this);
    }

    void set_activation(graph::OpType act_type) { act_op_ = nn::util::get_activation_op(act_type); }

    Param* param() override { return param_.get(); }
    Input* input() const { return input_; }

  protected:
    int out_offset_ = 0;
    FusedConcat* concat_ = nullptr;
    kernel::ActOp act_op_ = kernel::Linear{};

    SPtr<Param> param_;
    Input* input_;

    DenseMatrix& effective_weights() {
        return param_->has_factorizer() ? param_->factorizer().weights() : param_->weights().data();
    }

    Tensor& effective_output() { return concat_ ? concat_->output() : output_; }
};

struct SparseAffine : public SparseAffineBase {
    SparseAffine(SPtr<Param> param, Input* input)
        : SparseAffineBase(param, input) {}

    void forward() override {
        kernel::sparse_affine_fwd(
            effective_weights(),
            param_->biases().data(),
            effective_output().data(),
            input_->indices(),
            out_offset_,
            act_op_
        );
    }

    void backward() override {
        kernel::sparse_affine_bwd(
            param_->weights().grad(),
            param_->biases().grad(),
            effective_output(),
            input_->indices(),
            out_offset_,
            act_op_
        );
    }
};

struct SparseAffinePairwiseMul : public SparseAffineBase {
    SparseAffinePairwiseMul(SPtr<Param> param, Input* input)
        : SparseAffineBase(param, input, 2) {}

    void forward() override {
        kernel::sparse_affine_pairwise_mul_fwd(
            effective_weights(),
            param_->biases().data(),
            effective_output().data(),
            input_->indices(),
            out_offset_,
            act_op_
        );
    }

    void backward() override {
        kernel::sparse_affine_pairwise_mul_bwd(
            effective_weights(),
            param_->weights().grad(),
            param_->biases(),
            effective_output().grad(),
            input_->indices(),
            out_offset_,
            act_op_
        );
    }
};

} // namespace nn::op
