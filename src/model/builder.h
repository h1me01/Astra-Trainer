#pragma once

#include "../nn/include.h"

namespace model {

namespace helper {

template <typename T, typename... Args>
auto make(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

template <typename T, typename U>
auto dpc(U&& ptr) {
    return std::dynamic_pointer_cast<T>(std::forward<U>(ptr));
}

} // namespace helper

namespace save_format {

using Type = nn::SaveFormat::Type;

constexpr auto int8 = Type::int8;
constexpr auto int16 = Type::int16;
constexpr auto float32 = Type::float32;

} // namespace save_format

namespace param {

inline Ptr<nn::Param> create(int input_dim, int output_dim) {
    return helper::make<nn::Param>(input_dim, output_dim);
}

} // namespace param

namespace op {

class OpHandle {
  public:
    OpHandle(Ptr<nn::Operation> op)
        : op(op) {}

    OpHandle relu() { return set_activation<Activation::ReLU>(); }
    OpHandle clipped_relu() { return set_activation<Activation::ClippedReLU>(); }
    OpHandle sqr_clipped_relu() { return set_activation<Activation::SqrClippedReLU>(); }
    OpHandle sigmoid() { return set_activation<Activation::Sigmoid>(); }
    OpHandle select(Ptr<nn::SelectIndices> indices) { return OpHandle(helper::make<nn::Select>(op, indices)); }
    OpHandle pairwise_mul() { return OpHandle(helper::make<nn::PairwiseMul>(op)); }

    operator Ptr<nn::Operation>() const { return op; }

    Ptr<nn::Operation> get() const { return op; }

  private:
    Ptr<nn::Operation> op;

    template <Activation act_type>
    OpHandle set_activation() {
        if (op->get_name() == "concat") {
            auto concat_op = helper::dpc<nn::Concat>(op);
            // for concats who are fused we want them to not have an activation
            // since it would mess up the fusion, so seperate it
            if (concat_op->should_skip())
                return OpHandle(helper::make<nn::Activate>(op, act_type));
        }

        if (op->get_activation() != Activation::Linear)
            return OpHandle(helper::make<nn::Activate>(op, act_type));

        op->set_activation(act_type);
        return *this;
    }
};

class SparseAffineBuilder {
  public:
    SparseAffineBuilder(int input_dim, int output_dim)
        : params(helper::make<nn::Param>(input_dim, output_dim)) {}

    OpHandle operator()(Ptr<nn::Input> a) { return OpHandle(helper::make<nn::SparseAffine>(params, a)); }

    nn::SaveFormat& weights_format() { return params->weights_format(); }
    nn::SaveFormat& biases_format() { return params->biases_format(); }

    const nn::SaveFormat& weights_format() const { return params->weights_format(); }
    const nn::SaveFormat& biases_format() const { return params->biases_format(); }

    Ptr<nn::Param> get_param() { return params; }

  private:
    Ptr<nn::Param> params;
};

class AffineBuilder {
  public:
    AffineBuilder(int input_dim, int output_dim)
        : params(helper::make<nn::Param>(input_dim, output_dim)) {}

    OpHandle operator()(Ptr<nn::Operation> a) {
        if (a->get_output_dim() != params->get_input_dim())
            error("Affine input dimension does not match parameter input dimension!");
        return OpHandle(helper::make<nn::Affine>(params, a));
    }

    nn::SaveFormat& weights_format() { return params->weights_format(); }
    nn::SaveFormat& biases_format() { return params->biases_format(); }

    const nn::SaveFormat& weights_format() const { return params->weights_format(); }
    const nn::SaveFormat& biases_format() const { return params->biases_format(); }

    Ptr<nn::Param> get_param() { return params; }

  private:
    Ptr<nn::Param> params;
};

inline SparseAffineBuilder sparse_affine(int input_dim, int output_dim) {
    return SparseAffineBuilder(input_dim, output_dim);
}

inline AffineBuilder affine(int input_dim, int output_dim) {
    return AffineBuilder(input_dim, output_dim);
}

template <typename Fn>
inline Ptr<nn::SelectIndices> select_indices(int count, Fn&& fn) {
    return helper::make<nn::SelectIndices>(count, std::forward<Fn>(fn));
}

inline OpHandle concat(std::vector<Ptr<nn::Operation>> inputs) {
    auto output = OpHandle(helper::make<nn::Concat>(inputs));

    if (inputs.empty())
        return output;

    const auto& type = inputs[0]->get_name();
    for (const auto& input : inputs)
        if (input->get_name() != type)
            return output;

    if (type != "pairwise_mul" && type != "sparse_affine")
        return output;

    auto concat_op = helper::dpc<nn::Concat>(output.get());
    concat_op->set_skip();

    // special multi-layer specific fusion
    if (type == "pairwise_mul") {
        Activation act_type = inputs[0]->get_activation();

        // check if all inputs have the same activation and sparse_affine structure
        bool can_fuse = true;
        for (const auto& input : inputs) {
            auto pw_mul = helper::dpc<nn::PairwiseMul>(input);
            if (!pw_mul || input->get_activation() != act_type ||
                pw_mul->get_inputs()[0]->get_name() != "sparse_affine") {
                can_fuse = false;
                break;
            }
        }

        if (can_fuse) {
            for (auto& input : inputs) {
                auto pw_mul = helper::dpc<nn::PairwiseMul>(input);
                pw_mul->set_skip();
                if (auto sparse_aff = helper::dpc<nn::SparseAffine>(pw_mul->get_inputs()[0]))
                    sparse_aff->set_concat(concat_op, true);
            }

            if (act_type != Activation::Linear)
                return OpHandle(helper::make<nn::Activate>(output.get(), act_type));

            return output;
        }
    }

    // standard fusion
    for (auto& input : inputs) {
        if (type == "sparse_affine") {
            if (auto op = helper::dpc<nn::SparseAffine>(input))
                op->set_concat(concat_op);
        } else { // pairwise_mul
            if (auto op = helper::dpc<nn::PairwiseMul>(input))
                op->set_concat(concat_op);
        }
    }

    return output;
}

} // namespace op

namespace lr_sched {

inline Ptr<nn::LRScheduler> constant(float lr) {
    return helper::make<nn::Constant>(lr);
}

inline Ptr<nn::LRScheduler> step_decay(int step_size, float decay_factor) {
    return helper::make<nn::StepDecay>(step_size, decay_factor);
}

inline Ptr<nn::LRScheduler> cosine_annealing(int total_epochs, float initial_lr, float final_lr) {
    return helper::make<nn::CosineAnnealing>(total_epochs, initial_lr, final_lr);
}

} // namespace lr_sched

namespace optim {

class OptimHandle {
  public:
    OptimHandle(Ptr<nn::Optimizer> optim)
        : optim(optim) {}

    OptimHandle clamp(float min, float max) {
        optim->clamp(min, max);
        return *this;
    }

    operator Ptr<nn::Optimizer>() const { return optim; }

    Ptr<nn::Optimizer> get() const { return optim; }

  private:
    Ptr<nn::Optimizer> optim;
};

inline OptimHandle adam(float beta1, float beta2) {
    return OptimHandle(helper::make<nn::Adam>(beta1, beta2));
}

inline OptimHandle adamw(float beta1, float beta2, float decay) {
    return OptimHandle(helper::make<nn::Adam>(beta1, beta2, decay));
}

} // namespace optim

namespace loss {

inline Ptr<nn::Loss> mse(Activation act = Activation::Linear) {
    return helper::make<nn::MSE>(act);
}

inline Ptr<nn::Loss> mpe(float power, Activation act = Activation::Linear) {
    return helper::make<nn::MPE>(power, act);
}

} // namespace loss

} // namespace model
