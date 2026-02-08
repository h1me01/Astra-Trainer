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

    OpHandle crelu() {
        op->crelu();
        return *this;
    }

    OpHandle screlu() {
        op->screlu();
        return *this;
    }

    OpHandle sigmoid() {
        op->sigmoid();
        return *this;
    }

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

    OpHandle operator()(Ptr<nn::Input> a, Ptr<nn::Input> b) {
        return OpHandle(detail::make<nn::FeatureTransformer>(params, a, b));
    }

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

    OpHandle operator()(Ptr<nn::Operation> a) { return OpHandle(detail::make<nn::Affine>(params, a)); }

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

inline OpHandle select(Ptr<nn::Operation> s, Ptr<nn::SelectIndices> indices) {
    return OpHandle(detail::make<nn::Select>(s, indices));
}

inline OpHandle concat(Ptr<nn::Operation> a, Ptr<nn::Operation> b) {
    return OpHandle(detail::make<nn::Concat>(a, b));
}

inline OpHandle pairwise_mul(Ptr<nn::Operation> a) {
    return OpHandle(detail::make<nn::PairwiseMul>(a));
}

// output will be concatenation of the two inputs
inline OpHandle pairwise_mul(Ptr<nn::Operation> a, Ptr<nn::Operation> b) {
    return OpHandle(detail::make<nn::PairwiseMul>(a, b));
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

inline OptimHandle adam(float beta1, float beta2, float eps) {
    return OptimHandle(detail::make<nn::Adam>(beta1, beta2, eps));
}

inline OptimHandle adamw(float beta1, float beta2, float eps, float decay) {
    return OptimHandle(detail::make<nn::Adam>(beta1, beta2, eps, decay));
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
