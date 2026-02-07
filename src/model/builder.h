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

inline Ptr<nn::FeatureTransformer> feature_transformer(Ptr<nn::Param> params, Ptr<nn::Input> a) {
    return detail::make<nn::FeatureTransformer>(params, a);
}

// output will be concatenation of the two inputs
inline Ptr<nn::FeatureTransformer> feature_transformer(Ptr<nn::Param> params, Ptr<nn::Input> a, Ptr<nn::Input> b) {
    return detail::make<nn::FeatureTransformer>(params, a, b);
}

inline Ptr<nn::Affine> affine(Ptr<nn::Param> params, Ptr<nn::Operation> a) {
    return detail::make<nn::Affine>(params, a);
}

template <typename Fn>
inline Ptr<nn::SelectIndices> select_indices(int count, Fn&& fn) {
    return detail::make<nn::SelectIndices>(count, std::forward<Fn>(fn));
}

inline Ptr<nn::Select> select(Ptr<nn::Operation> s, Ptr<nn::SelectIndices> indices) {
    return detail::make<nn::Select>(s, indices);
}

inline Ptr<nn::Concat> concat(Ptr<nn::Operation> a, Ptr<nn::Operation> b) {
    return detail::make<nn::Concat>(a, b);
}

inline Ptr<nn::PairwiseMul> pairwise_mul(Ptr<nn::Operation> a) {
    return detail::make<nn::PairwiseMul>(a);
}

// output will be concatenation of the two inputs
inline Ptr<nn::PairwiseMul> pairwise_mul(Ptr<nn::Operation> a, Ptr<nn::Operation> b) {
    return detail::make<nn::PairwiseMul>(a, b);
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

inline Ptr<nn::Optimizer> adam(float beta1, float beta2, float eps) {
    return detail::make<nn::Adam>(beta1, beta2, eps);
}

inline Ptr<nn::Optimizer> adamw(float beta1, float beta2, float eps, float decay) {
    return detail::make<nn::Adam>(beta1, beta2, eps, decay);
}

} // namespace optim

namespace loss {

inline Ptr<nn::Loss> mse() {
    return detail::make<nn::MSE>();
}

inline Ptr<nn::Loss> mpe(float power) {
    return detail::make<nn::MPE>(power);
}

} // namespace loss

} // namespace model
