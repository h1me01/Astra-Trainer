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

    OpType op_type() { return op_type_; }

    int output_dim() { return output_dim_; }

    std::string op_type_as_str() { return op_type_str(op_type_); }

    std::vector<SPtr<Node>>& inputs() { return inputs_; }
    const std::vector<SPtr<Node>>& inputs() const { return inputs_; }

  protected:
    OpType op_type_;
    int output_dim_;
    std::vector<SPtr<Node>> inputs_;
};

struct InputNode : public Node {
    InputNode(int output_dim)
        : Node(OpType::Input, output_dim, {}) {}
};

class SparseAffineNode : public Node {
  public:
    SparseAffineNode(SPtr<Param> param, SPtr<InputNode> input)
        : Node(OpType::SparseAffine, param->output_dim(), {input}),
          param_(param) {}

    SPtr<Param> param() { return param_; }

    void set_pairwise_fused() { pairwise_fused_ = true; }
    bool is_pairwise_fused() const { return pairwise_fused_; }

    void set_activation(OpType act_type) { this->act_type = act_type; }
    OpType activation() const { return act_type; }

  private:
    bool pairwise_fused_ = false;
    OpType act_type = OpType::None;

    SPtr<Param> param_;
};

class AffineNode : public Node {
  public:
    AffineNode(SPtr<Param> param, SPtr<Node> input)
        : Node(OpType::Affine, param->output_dim(), {input}),
          param_(param) {

        const int prev_dim = input->output_dim();
        if (param->input_dim() != prev_dim) {
            error(
                "Graph: Affine input dim mismatch! Expected " + std::to_string(param->input_dim()) + " got " +
                std::to_string(prev_dim)
            );
        }
    }

    SPtr<Param> param() { return param_; }

  private:
    SPtr<Param> param_;
};

class ConcatNode : public Node {
  public:
    ConcatNode(const std::vector<SPtr<Node>> inputs)
        : Node(OpType::Concat, calc_output_dim(inputs), inputs) {
        if (inputs.size() < 2)
            error("Graph: Concat must have at least 2 inputs!");
    }

    void set_fused() { fused_ = true; }
    bool is_fused() const { return fused_; }

  private:
    bool fused_ = false;

    int calc_output_dim(const std::vector<SPtr<Node>> inputs) {
        int dim = 0;
        for (const auto in : inputs)
            dim += in->output_dim();
        return dim;
    }
};

class SelectNode : public Node {
  public:
    SelectNode(SPtr<Node> input, SPtr<SelectIndices> indices)
        : Node(OpType::Select, input->output_dim() / indices->partitions_size(), {input}),
          indices_(indices) {
        if ((input->output_dim() % indices->partitions_size()) != 0)
            error("Graph: Select input dim must be divisable by the number of partitions!");
    }

    SPtr<SelectIndices> indices() { return indices_; }

  private:
    SPtr<SelectIndices> indices_;
};

struct PairwiseMulNode : public Node {
    PairwiseMulNode(SPtr<Node> input)
        : Node(OpType::PairwiseMul, input->output_dim() / 2, {input}) {
        if ((input->output_dim() % 2) != 0)
            error("Graph: PairwiseMul input dim must be even!");
    }
};

template <typename Op>
class UnaryNode : public Node {
  public:
    UnaryNode(OpType op_type, SPtr<Node> input, Op op)
        : Node(op_type, input->output_dim(), {input}),
          op_(op) {

        if (!is_unary(op_type))
            error("Graph: invalid unary type!");
    }

    Op op() const { return op_; }

  private:
    Op op_;
};

template <typename Op>
class BinaryNode : public Node {
  public:
    BinaryNode(OpType op_type, SPtr<Node> input1, SPtr<Node> input2, Op op)
        : Node(op_type, input1->output_dim(), {input1, input2}),
          op_(op) {

        if (!is_elemwise(op_type))
            error("Graph: invalid elemwise type!");
        if (input1->output_dim() != input2->output_dim())
            error("Graph: Elemwise inputs must have the same output dimension!");
    }

    Op op() const { return op_; }

  private:
    Op op_;
};

} // namespace nn::graph
