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

    OpHandle relu() {
        op->set_activation(Activation::ReLU);
        return *this;
    }

    OpHandle clamped_relu() {
        op->set_activation(Activation::ClampedReLU);
        return *this;
    }

    OpHandle squared_clamped_relu() {
        op->set_activation(Activation::SquaredClampedReLU);
        return *this;
    }

    OpHandle sigmoid() {
        op->set_activation(Activation::Sigmoid);
        return *this;
    }

    OpHandle select(Ptr<nn::SelectIndices> indices) { return OpHandle(helper::make<nn::Select>(op, indices)); }

    OpHandle pairwise_mul() {
        // fuse sparse_affine + pairwise_mul into single operation
        if (op->get_name() == "sparse_affine") {
            auto sa = helper::dpc<nn::SparseAffine>(op);
            if (sa) {
                auto inputs = sa->get_inputs_ft();
                if (inputs.size() == 1) {
                    auto fused = helper::make<nn::SparseAffine>(sa->get_param(), inputs[0], true);
                    fused->set_activation(op->get_activation());
                    return OpHandle(fused);
                }
            }
        }

        return OpHandle(helper::make<nn::PairwiseMul>(op));
    }

    operator Ptr<nn::Operation>() const { return op; }

    Ptr<nn::Operation> get() const { return op; }

  private:
    Ptr<nn::Operation> op;
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

inline OpHandle concat(Ptr<nn::Operation> a, Ptr<nn::Operation> b) {
    auto try_fuse_sparse_affine = [](auto sa_a, auto sa_b, bool pairwise) -> OpHandle {
        if (
            sa_a &&                                          //
            sa_b &&                                          //
            sa_a->get_inputs_ft().size() == 1 &&             //
            sa_b->get_inputs_ft().size() == 1 &&             //
            sa_a->get_param() == sa_b->get_param() &&        //
            sa_a->get_activation() == sa_b->get_activation() //
        ) {
            auto fused = helper::make<nn::SparseAffine>(
                sa_a->get_param(), sa_a->get_inputs_ft()[0], sa_b->get_inputs_ft()[0], pairwise
            );

            fused->set_activation(sa_a->get_activation());
            return OpHandle(fused);
        }

        return OpHandle(nullptr);
    };

    auto a_name = a->get_name();
    auto b_name = b->get_name();

    if (a_name == "pairwise_mul" && b_name == "pairwise_mul") {
        auto pwm_a = helper::dpc<nn::PairwiseMul>(a);
        auto pwm_b = helper::dpc<nn::PairwiseMul>(b);

        if (
            pwm_a &&                                           //
            pwm_b &&                                           //
            pwm_a->get_inputs().size() == 1 &&                 //
            pwm_b->get_inputs().size() == 1 &&                 //
            pwm_a->get_activation() == pwm_b->get_activation() //
        ) {
            auto result = try_fuse_sparse_affine(
                helper::dpc<nn::SparseAffine>(pwm_a->get_inputs()[0]),
                helper::dpc<nn::SparseAffine>(pwm_b->get_inputs()[0]),
                true
            );
            if (result.get() != nullptr)
                return result;

            // fall back to regular pairwise mul fusion
            auto fused = helper::make<nn::PairwiseMul>(pwm_a->get_inputs()[0], pwm_b->get_inputs()[0]);
            fused->set_activation(a->get_activation());

            return OpHandle(fused);
        }
    }

    if (a_name == "sparse_affine_pairwise_mul" && b_name == "sparse_affine_pairwise_mul") {
        auto result = try_fuse_sparse_affine(helper::dpc<nn::SparseAffine>(a), helper::dpc<nn::SparseAffine>(b), true);
        if (result.get() != nullptr)
            return result;
    }

    if (a_name == "sparse_affine" && b_name == "sparse_affine") {
        auto result = try_fuse_sparse_affine(helper::dpc<nn::SparseAffine>(a), helper::dpc<nn::SparseAffine>(b), false);
        if (result.get() != nullptr)
            return result;
    }

    // no fusion
    return OpHandle(helper::make<nn::Concat>(a, b));
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
