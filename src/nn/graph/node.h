#pragma once

#include "../ops/include.h"
#include "../param/param.h"
#include "common.h"

namespace nn::graph {

using namespace nn::param;
using namespace nn::op;

class Node {
  public:
    Node(OpType op_type, int output_dim, std::vector<Node*> inputs)
        : op_type(op_type),
          output_dim(output_dim),
          inputs(inputs) {
        for (const auto in : inputs)
            if (in == nullptr)
                error("Graph: inputs cannot be null!");
    }

    virtual ~Node() = default;

    OpType get_op_type() { return op_type; }

    int get_output_dim() { return output_dim; }

    std::string get_op_type_str() { return op_type_str(op_type); }

    std::vector<Node*>& get_inputs() { return inputs; }
    const std::vector<Node*>& get_inputs() const { return inputs; }

    void set_activation(OpType act_type) {
        CHECK(is_activation(act_type));
        this->act_type = act_type;
    }

    OpType get_activation() const { return act_type; }

  protected:
    OpType op_type;
    int output_dim;
    OpType act_type = OpType::None;
    std::vector<Node*> inputs;
};

struct InputNode : public Node {
    InputNode(int output_dim)
        : Node(OpType::Input, output_dim, {}) {}
};

class SparseAffineNode : public Node {
  public:
    SparseAffineNode(Param* param, InputNode* input)
        : Node(OpType::SparseAffine, param->get_output_dim(), {input}),
          param(param) {}

    Param* get_param() { return param; }

    void set_pairwise_fused() { pairwise_fused = true; }
    bool is_pairwise_fused() const { return pairwise_fused; }

  private:
    bool pairwise_fused = false;
    Param* param;
};

class AffineNode : public Node {
  public:
    AffineNode(Param* param, Node* input)
        : Node(OpType::Affine, param->get_output_dim(), {input}),
          param(param) {

        const int prev_dim = input->get_output_dim();
        if (param->get_input_dim() != prev_dim) {
            error(
                "Graph: Affine input dim mismatch! Expected " + std::to_string(param->get_input_dim()) + " got " +
                std::to_string(prev_dim)
            );
        }
    }

    Param* get_param() { return param; }

  private:
    Param* param;
};

class ConcatNode : public Node {
  public:
    ConcatNode(const std::vector<Node*> inputs)
        : Node(OpType::Concat, get_output_dim(inputs), inputs) {
        if (inputs.size() < 2)
            error("Graph: Concat must have at least 2 inputs!");
    }

    void set_fused() { fused = true; }
    bool is_fused() const { return fused; }

  private:
    bool fused = false;

    int get_output_dim(const std::vector<Node*> inputs) {
        int dim = 0;
        for (const auto in : inputs)
            dim += in->get_output_dim();
        return dim;
    }
};

class SelectNode : public Node {
  public:
    SelectNode(Node* input, SelectIndices* indices)
        : Node(OpType::Select, input->get_output_dim() / indices->partitions_size(), {input}),
          indices(indices) {
        if ((input->get_output_dim() % indices->partitions_size()) != 0)
            error("Graph: Select input dim must be divisable by the number of partitions!");
    }

    SelectIndices* get_indices() { return indices; }

  private:
    SelectIndices* indices;
};

struct PairwiseMulNode : public Node {
    PairwiseMulNode(Node* input)
        : Node(OpType::PairwiseMul, input->get_output_dim() / 2, {input}) {
        if ((input->get_output_dim() % 2) != 0)
            error("Graph: PairwiseMul input dim must be even!");
    }
};

struct ActivationNode : public Node {
    ActivationNode(OpType op_type, Node* input)
        : Node(op_type, input->get_output_dim(), {input}) {}
};

} // namespace nn::graph
