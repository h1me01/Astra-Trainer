#pragma once

#include "../nn/include.h"

namespace model {

namespace detail {
template <typename T, typename... Args>
auto make(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}
} // namespace detail

namespace save_format {

using Type = nn::SaveFormat::Type;

constexpr auto int8 = Type::int8;
constexpr auto int16 = Type::int16;
constexpr auto float32 = Type::float32;

} // namespace save_format

namespace param {

inline Ptr<nn::Param> create(int input_dim, int output_dim) {
    return detail::make<nn::Param>(input_dim, output_dim);
}

} // namespace param

namespace op {

class OpHandle {
  public:
    OpHandle(Ptr<nn::Operation> op)
        : op(op) {}

    OpHandle relu() {
        op->relu();
        return *this;
    }

    OpHandle clamped_relu() {
        op->clamped_relu();
        return *this;
    }

    OpHandle squared_clamped_relu() {
        op->squared_clamped_relu();
        return *this;
    }

    OpHandle sigmoid() {
        op->sigmoid();
        return *this;
    }

    OpHandle select(Ptr<nn::SelectIndices> indices) { return OpHandle(detail::make<nn::Select>(op, indices)); }

    OpHandle pairwise_mul() { return OpHandle(detail::make<nn::PairwiseMul>(op)); }

    operator Ptr<nn::Operation>() const { return op; }

    Ptr<nn::Operation> get() const { return op; }

  private:
    Ptr<nn::Operation> op;
};

class FeatureTransformerBuilder {
  public:
    FeatureTransformerBuilder(int input_dim, int output_dim)
        : params(detail::make<nn::Param>(input_dim, output_dim)) {}

    OpHandle operator()(Ptr<nn::Input> a) { return OpHandle(detail::make<nn::FeatureTransformer>(params, a)); }

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
        : params(detail::make<nn::Param>(input_dim, output_dim)) {}

    OpHandle operator()(Ptr<nn::Operation> a) {
        if (a->get_output_dim() != params->get_input_dim())
            error("Affine input dimension does not match parameter input dimension!");
        return OpHandle(detail::make<nn::Affine>(params, a));
    }

    nn::SaveFormat& weights_format() { return params->weights_format(); }
    nn::SaveFormat& biases_format() { return params->biases_format(); }

    const nn::SaveFormat& weights_format() const { return params->weights_format(); }
    const nn::SaveFormat& biases_format() const { return params->biases_format(); }

    Ptr<nn::Param> get_param() { return params; }

  private:
    Ptr<nn::Param> params;
};

inline FeatureTransformerBuilder feature_transformer(int input_dim, int output_dim) {
    return FeatureTransformerBuilder(input_dim, output_dim);
}

inline AffineBuilder affine(int input_dim, int output_dim) {
    return AffineBuilder(input_dim, output_dim);
}

template <typename Fn>
inline Ptr<nn::SelectIndices> select_indices(int count, Fn&& fn) {
    return detail::make<nn::SelectIndices>(count, std::forward<Fn>(fn));
}

inline OpHandle concat(Ptr<nn::Operation> a, Ptr<nn::Operation> b) {
    if (a->get_name() == "pairwise_mul" && b->get_name() == "pairwise_mul") {
        auto pwm_a = std::dynamic_pointer_cast<nn::PairwiseMul>(a);
        auto pwm_b = std::dynamic_pointer_cast<nn::PairwiseMul>(b);

        if (pwm_a && pwm_b) {
            auto inputs_a = pwm_a->get_inputs();
            auto inputs_b = pwm_b->get_inputs();

            if (inputs_a.size() == 1 && inputs_b.size() == 1 && a->get_activation() == b->get_activation()) {
                auto fused = detail::make<nn::PairwiseMul>(inputs_a[0], inputs_b[0]);

                Activation act = a->get_activation();
                if (act == Activation::ClampedReLU)
                    fused->clamped_relu();
                else if (act == Activation::ReLU)
                    fused->relu();
                else if (act == Activation::SquaredClampedReLU)
                    fused->squared_clamped_relu();
                else if (act == Activation::Sigmoid)
                    fused->sigmoid();

                return OpHandle(fused);
            }
        }
    }

    if (a->get_name() == "feature_transformer" && b->get_name() == "feature_transformer") {
        auto ft_a = std::dynamic_pointer_cast<nn::FeatureTransformer>(a);
        auto ft_b = std::dynamic_pointer_cast<nn::FeatureTransformer>(b);

        if (ft_a && ft_b) {
            auto param_a = ft_a->get_param();
            auto param_b = ft_b->get_param();

            if (param_a == param_b && a->get_activation() == b->get_activation()) {
                auto inputs_a = ft_a->get_inputs_ft();
                auto inputs_b = ft_b->get_inputs_ft();

                if (inputs_a.size() == 1 && inputs_b.size() == 1) {
                    auto fused = detail::make<nn::FeatureTransformer>(param_a, inputs_a[0], inputs_b[0]);

                    Activation act = a->get_activation();
                    if (act == Activation::ClampedReLU)
                        fused->clamped_relu();
                    else if (act == Activation::ReLU)
                        fused->relu();
                    else if (act == Activation::SquaredClampedReLU)
                        fused->squared_clamped_relu();
                    else if (act == Activation::Sigmoid)
                        fused->sigmoid();

                    return OpHandle(fused);
                }
            }
        }
    }

    return OpHandle(detail::make<nn::Concat>(a, b));
}

} // namespace op

namespace lr_sched {

inline Ptr<nn::LRScheduler> constant(float lr) {
    return detail::make<nn::Constant>(lr);
}

inline Ptr<nn::LRScheduler> step_decay(int step_size, float decay_factor) {
    return detail::make<nn::StepDecay>(step_size, decay_factor);
}

inline Ptr<nn::LRScheduler> cosine_annealing(int total_epochs, float initial_lr, float final_lr) {
    return detail::make<nn::CosineAnnealing>(total_epochs, initial_lr, final_lr);
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
    return OptimHandle(detail::make<nn::Adam>(beta1, beta2));
}

inline OptimHandle adamw(float beta1, float beta2, float decay) {
    return OptimHandle(detail::make<nn::Adam>(beta1, beta2, decay));
}

} // namespace optim

namespace loss {

inline Ptr<nn::Loss> mse(Activation act = Activation::Linear) {
    return detail::make<nn::MSE>(act);
}

inline Ptr<nn::Loss> mpe(float power, Activation act = Activation::Linear) {
    return detail::make<nn::MPE>(power, act);
}

} // namespace loss

} // namespace model
