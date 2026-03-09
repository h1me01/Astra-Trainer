#pragma once

#include "../misc.h"
#include "common.h"

namespace model {

namespace ng = nn::graph;
namespace np = nn::param;

namespace save_format {

using Type = nn::SaveFormat::Type;

constexpr auto int8 = Type::int8;
constexpr auto int16 = Type::int16;
constexpr auto float32 = Type::float32;

} // namespace save_format

inline int num_buckets(const std::array<int, 64>& bucket_map) {
    int max_bucket = 0;
    for (int b : bucket_map)
        max_bucket = std::max(max_bucket, b);
    return max_bucket + 1;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

class NodeHandle {
  public:
    NodeHandle(SPtr<ng::Node> node)
        : node(node) {}

    NodeHandle relu() { return make_activation(ng::OpType::ReLU); }
    NodeHandle clipped_relu() { return make_activation(ng::OpType::ClippedReLU); }
    NodeHandle sqr_clipped_relu() { return make_activation(ng::OpType::SqrClippedReLU); }
    NodeHandle sigmoid() { return make_activation(ng::OpType::Sigmoid); }
    NodeHandle select(SelectIndices indices) { return NodeHandle(std::make_shared<ng::SelectNode>(node, indices)); }
    NodeHandle pairwise_mul() { return NodeHandle(std::make_shared<ng::PairwiseMulNode>(node)); }

    operator SPtr<ng::Node>() const { return node; }
    SPtr<ng::Node> get() const { return node; }

  private:
    SPtr<ng::Node> node;

    NodeHandle make_activation(ng::OpType op_type) {
        return NodeHandle(std::make_shared<ng::ActivationNode>(op_type, node));
    }
};

class SparseAffineBuilder {
  public:
    SparseAffineBuilder(int input_dim, int output_dim)
        : param(std::make_shared<np::Param>(input_dim, output_dim)) {}

    SparseAffineBuilder& factorized(int block_size) {
        if (param->has_factorizer())
            error("SparseAffineBuilder: Factorizer already exists for this layer!");

        param->create_factorizer(block_size);
        return *this;
    }

    NodeHandle operator()(SPtr<ng::InputNode> a) {
        return NodeHandle(std::make_shared<ng::SparseAffineNode>(param, a));
    }

    Tensor& get_weights() { return param->get_weights(); }
    Tensor& get_biases() { return param->get_biases(); }

    np::SaveFormat& weights_format() { return param->weights_format(); }
    np::SaveFormat& biases_format() { return param->biases_format(); }

  private:
    SPtr<np::Param> param;
};

class AffineBuilder {
  public:
    AffineBuilder(int input_dim, int output_dim)
        : param(std::make_shared<np::Param>(input_dim, output_dim)) {}

    NodeHandle operator()(SPtr<ng::Node> a) { return NodeHandle(std::make_shared<ng::AffineNode>(param, a)); }

    Tensor& get_weights() { return param->get_weights(); }
    Tensor& get_biases() { return param->get_biases(); }

    np::SaveFormat& weights_format() { return param->weights_format(); }
    np::SaveFormat& biases_format() { return param->biases_format(); }

  private:
    SPtr<np::Param> param;
};

namespace graph {

inline InputNode create_input(int size) {
    return std::make_shared<ng::InputNode>(size);
}

inline SparseAffineBuilder sparse_affine(int input_dim, int output_dim) {
    return SparseAffineBuilder(input_dim, output_dim);
}

inline AffineBuilder affine(int input_dim, int output_dim) {
    return AffineBuilder(input_dim, output_dim);
}

template <typename Fn>
inline SelectIndices select_index_fn(int count, Fn&& fn) {
    return std::make_shared<nn::op::SelectIndices>(count, std::forward<Fn>(fn));
}

inline NodeHandle concat(std::vector<NodeHandle> inputs) {
    std::vector<SPtr<ng::Node>> nodes;
    for (auto& h : inputs)
        nodes.push_back(h.get());
    return NodeHandle(std::make_shared<ng::ConcatNode>(nodes));
}

} // namespace graph

namespace lr_sched {

inline LRScheduler constant(float lr) {
    return std::make_unique<nn::lr_sched::Constant>(lr);
}

inline LRScheduler step_decay(float lr, float gamma, int step_size) {
    return std::make_unique<nn::lr_sched::StepDecay>(lr, gamma, step_size);
}

inline LRScheduler cosine_annealing(float start_lr, float final_lr, int max_epochs) {
    return std::make_unique<nn::lr_sched::CosineAnnealing>(start_lr, final_lr, max_epochs);
}

} // namespace lr_sched

namespace wdl_sched {

inline WDLScheduler constant(float val) {
    return std::make_unique<nn::wdl_sched::Constant>(val);
}

inline WDLScheduler linear(float start_val, float final_val, int max_epochs) {
    return std::make_unique<nn::wdl_sched::Linear>(start_val, final_val, max_epochs);
}

} // namespace wdl_sched

class OptimHandle {
  public:
    explicit OptimHandle(Optimizer optim)
        : optim(std::move(optim)) {}

    OptimHandle& clamp_params(float min, float max) & {
        optim->clamp_params(min, max);
        return *this;
    }

    OptimHandle&& clamp_params(float min, float max) && {
        optim->clamp_params(min, max);
        return std::move(*this);
    }

    Optimizer take() { return std::move(optim); }

  private:
    Optimizer optim;
};

namespace optim {

inline OptimHandle adam(float beta1, float beta2) {
    return OptimHandle(std::make_unique<nn::optim::Adam>(beta1, beta2));
}

inline OptimHandle adamw(float beta1, float beta2, float decay) {
    return OptimHandle(std::make_unique<nn::optim::Adam>(beta1, beta2, decay));
}

} // namespace optim

namespace loss {

inline Loss mse(ActivationType act = ActivationType::Linear) {
    return std::make_unique<nn::loss::MPE>(2.0, act);
}

inline Loss mpe(float power, ActivationType act = ActivationType::Linear) {
    return std::make_unique<nn::loss::MPE>(power, act);
}

} // namespace loss

namespace dataloader {

inline Dataloader create(
    int thread_count,
    std::vector<std::string> filenames,
    std::function<bool(const TrainingDataEntry&)> skip_predicate = nullptr
) {
    return std::make_unique<nn::dataloader::Dataloader>(thread_count, filenames, skip_predicate);
}

} // namespace dataloader

} // namespace model
