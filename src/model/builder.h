#pragma once

#include "common.h"

namespace model {

namespace save_format {

using Type = nn::SaveFormat::Type;

constexpr auto int8 = Type::int8;
constexpr auto int16 = Type::int16;
constexpr auto float32 = Type::float32;

} // namespace save_format

namespace param {

inline SPtr<nn::Param> create(int input_dim, int output_dim) {
    return std::make_shared<nn::Param>(input_dim, output_dim);
}

} // namespace param

namespace op {

class OpHandle {
  public:
    OpHandle(Operation op)
        : op(op) {}

    OpHandle relu() { return set_activation<Activation::ReLU>(); }
    OpHandle clipped_relu() { return set_activation<Activation::ClippedReLU>(); }
    OpHandle sqr_clipped_relu() { return set_activation<Activation::SqrClippedReLU>(); }
    OpHandle sigmoid() { return set_activation<Activation::Sigmoid>(); }
    OpHandle select(SelectIndices indices) { return OpHandle(std::make_shared<nn::Select>(op, indices)); }

    OpHandle pairwise_mul() {
        if (auto sparse_aff = dpc<nn::SparseAffine>(op)) {
            if (!sparse_aff->is_pairwise_fused()) {
                sparse_aff->set_pairwise_fused();
                return OpHandle(sparse_aff);
            }
        }
        return OpHandle(std::make_shared<nn::PairwiseMul>(op));
    }

    operator Operation() const { return op; }

    Operation get() const { return op; }

  private:
    Operation op;

    template <Activation act_type>
    OpHandle set_activation() {
        // if activation already exists, separate
        bool needs_separate = (op->get_activation() != Activation::Linear);

        if (!needs_separate) {
            // for fused concats we separate the activation
            if (auto concat = dpc<nn::Concat>(op))
                needs_separate = concat->should_skip();
            // for sparse affine with pairwise fusion we seperate the activation
            else if (auto sparse = dpc<nn::SparseAffine>(op))
                needs_separate = sparse->is_pairwise_fused();
        }

        if (needs_separate)
            return OpHandle(std::make_shared<nn::Activate>(op, act_type));

        op->set_activation(act_type);
        return *this;
    }
};

class SparseAffineBuilder {
  public:
    SparseAffineBuilder(int input_dim, int output_dim)
        : params(std::make_shared<nn::Param>(input_dim, output_dim)) {}

    OpHandle operator()(Input a) { return OpHandle(std::make_shared<nn::SparseAffine>(params, a)); }

    Tensor& get_weights() { return params->get_weights(); }
    Tensor& get_biases() { return params->get_biases(); }

    nn::SaveFormat& weights_format() { return params->weights_format(); }
    nn::SaveFormat& biases_format() { return params->biases_format(); }

    SPtr<nn::Param> get_param() { return params; }

  private:
    SPtr<nn::Param> params;
};

class AffineBuilder {
  public:
    AffineBuilder(int input_dim, int output_dim)
        : params(std::make_shared<nn::Param>(input_dim, output_dim)) {}

    OpHandle operator()(Operation a) { return OpHandle(std::make_shared<nn::Affine>(params, a)); }

    Tensor& get_weights() { return params->get_weights(); }
    Tensor& get_biases() { return params->get_biases(); }

    nn::SaveFormat& weights_format() { return params->weights_format(); }
    nn::SaveFormat& biases_format() { return params->biases_format(); }

    SPtr<nn::Param> get_param() { return params; }

  private:
    SPtr<nn::Param> params;
};

inline SparseAffineBuilder sparse_affine(int input_dim, int output_dim) {
    return SparseAffineBuilder(input_dim, output_dim);
}

inline AffineBuilder affine(int input_dim, int output_dim) {
    return AffineBuilder(input_dim, output_dim);
}

template <typename Fn>
inline SelectIndices select_indices(int count, Fn&& fn) {
    return std::make_shared<nn::SelectIndices>(count, std::forward<Fn>(fn));
}

inline OpHandle concat(std::vector<Operation> inputs) {
    auto output = OpHandle(std::make_shared<nn::Concat>(inputs));

    for (const auto& input : inputs)
        if (input->get_name() != "sparse_affine" && input->get_name() != "pairwise_mul")
            return output;

    auto concat_op = dpc<nn::Concat>(output.get());
    concat_op->set_skip();

    // currently only sparse affine and pairwise mul fusion is supported
    // when needed, adding future fusions should be trivial
    for (auto& input : inputs) {
        if (auto op = dpc<nn::SparseAffine>(input))
            op->set_concat(concat_op);
        else if (auto op = dpc<nn::PairwiseMul>(input))
            op->set_concat(concat_op);
        else
            ASSERT(false);
    }

    return output;
}

} // namespace op

namespace lr_sched {

inline LRScheduler constant(float lr) {
    return std::make_shared<nn::lr_sched::Constant>(lr);
}

inline LRScheduler step_decay(float lr, float gamma, int step_size) {
    return std::make_shared<nn::lr_sched::StepDecay>(lr, gamma, step_size);
}

inline LRScheduler cosine_annealing(float start_lr, float final_lr, int max_epochs) {
    return std::make_shared<nn::lr_sched::CosineAnnealing>(start_lr, final_lr, max_epochs);
}

} // namespace lr_sched

namespace wdl_sched {

inline WDLScheduler constant(float val) {
    return std::make_shared<nn::wdl_sched::Constant>(val);
}

inline WDLScheduler linear(float start_val, float final_val, int max_epochs) {
    return std::make_shared<nn::wdl_sched::Linear>(start_val, final_val, max_epochs);
}

} // namespace wdl_sched

namespace optim {

class OptimHandle {
  public:
    OptimHandle(Optimizer optim)
        : optim(optim) {}

    OptimHandle clamp(float min, float max) {
        optim->clamp(min, max);
        return *this;
    }

    operator Optimizer() const { return optim; }

    Optimizer get() const { return optim; }

  private:
    Optimizer optim;
};

inline OptimHandle adam(float beta1, float beta2) {
    return OptimHandle(std::make_shared<nn::Adam>(beta1, beta2));
}

inline OptimHandle adamw(float beta1, float beta2, float decay) {
    return OptimHandle(std::make_shared<nn::Adam>(beta1, beta2, decay));
}

} // namespace optim

namespace loss {

inline Loss mse(Activation act = Activation::Linear) {
    return std::make_shared<nn::loss::MPE>(2.0, act);
}

inline Loss mpe(float power, Activation act = Activation::Linear) {
    return std::make_shared<nn::loss::MPE>(power, act);
}

} // namespace loss

} // namespace model
