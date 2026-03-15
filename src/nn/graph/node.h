#pragma once

#include "../ops/include.h"
#include "../param/param.h"
#include "common.h"

namespace nn::graph {

using namespace nn::param;
using namespace nn::op;

class Node {
  public:
    Node(OpType op_type, int output_dim, std::vector<SPtr<Node>> inputs)
        : op_type_(op_type),
          output_dim_(output_dim),
          inputs_(inputs) {
        for (const auto in : inputs)
            if (!in)
                error("Graph: inputs cannot be null!");
    }

    virtual ~Node() = default;

    OpType get_op_type() { return op_type_; }

    int get_output_dim() { return output_dim_; }

    std::string get_op_type_str() { return op_type_str(op_type_); }

    std::vector<SPtr<Node>>& get_inputs() { return inputs_; }
    const std::vector<SPtr<Node>>& get_inputs() const { return inputs_; }

    void set_activation(OpType act_type) {
        CHECK(is_activation(act_type));
        this->act_type_ = act_type;
    }

    OpType get_activation() const { return act_type_; }

  protected:
    OpType op_type_;
    int output_dim_;
    OpType act_type_ = OpType::None;
    std::vector<SPtr<Node>> inputs_;
};

struct InputNode : public Node {
    InputNode(int output_dim)
        : Node(OpType::Input, output_dim, {}) {}
};

class SparseAffineNode : public Node {
  public:
    SparseAffineNode(SPtr<Param> param, SPtr<InputNode> input)
        : Node(OpType::SparseAffine, param->get_output_dim(), {input}),
          param_(param) {}

    SPtr<Param> get_param() { return param_; }

    void set_pairwise_fused() { pairwise_fused_ = true; }
    bool is_pairwise_fused() const { return pairwise_fused_; }

  private:
    bool pairwise_fused_ = false;
    SPtr<Param> param_;
};

class AffineNode : public Node {
  public:
    AffineNode(SPtr<Param> param, SPtr<Node> input)
        : Node(OpType::Affine, param->get_output_dim(), {input}),
          param_(param) {

        const int prev_dim = input->get_output_dim();
        if (param->get_input_dim() != prev_dim) {
            error(
                "Graph: Affine input dim mismatch! Expected " + std::to_string(param->get_input_dim()) + " got " +
                std::to_string(prev_dim)
            );
        }
    }

    SPtr<Param> get_param() { return param_; }

  private:
    SPtr<Param> param_;
};

class ConcatNode : public Node {
  public:
    ConcatNode(const std::vector<SPtr<Node>> inputs)
        : Node(OpType::Concat, get_output_dim(inputs), inputs) {
        if (inputs.size() < 2)
            error("Graph: Concat must have at least 2 inputs!");
    }

    void set_fused() { fused_ = true; }
    bool is_fused() const { return fused_; }

  private:
    bool fused_ = false;

    int get_output_dim(const std::vector<SPtr<Node>> inputs) {
        int dim = 0;
        for (const auto in : inputs)
            dim += in->get_output_dim();
        return dim;
    }
};

class SelectNode : public Node {
  public:
    SelectNode(SPtr<Node> input, SPtr<SelectIndices> indices)
        : Node(OpType::Select, input->get_output_dim() / indices->partitions_size(), {input}),
          indices_(indices) {
        if ((input->get_output_dim() % indices->partitions_size()) != 0)
            error("Graph: Select input dim must be divisable by the number of partitions!");
    }

    SPtr<SelectIndices> get_indices() { return indices_; }

  private:
    SPtr<SelectIndices> indices_;
};

struct PairwiseMulNode : public Node {
    PairwiseMulNode(SPtr<Node> input)
        : Node(OpType::PairwiseMul, input->get_output_dim() / 2, {input}) {
        if ((input->get_output_dim() % 2) != 0)
            error("Graph: PairwiseMul input dim must be even!");
    }
};

struct ActivationNode : public Node {
    ActivationNode(OpType op_type, SPtr<Node> input)
        : Node(op_type, input->get_output_dim(), {input}) {}
};

} // namespace nn::graph
