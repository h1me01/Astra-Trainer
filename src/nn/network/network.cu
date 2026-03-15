#include "network.h"

namespace nn {

// helper

template <typename Kernel>
UPtr<Operation> make_unary(Node* node, Operation* input) {
    auto* n = dynamic_cast<UnaryNode<Kernel>*>(node);
    CHECK(n);
    return std::make_unique<op::Unary<Kernel>>(input, n->op());
}

template <typename Kernel>
UPtr<Operation> make_binary(Node* node, Operation* a, Operation* b) {
    auto* n = dynamic_cast<BinaryNode<Kernel>*>(node);
    CHECK(n);
    return std::make_unique<op::Binary<Kernel>>(a, b, n->op());
}

// Network

void Network::init_operations(const Graph& graph) {
    std::unordered_map<Node*, Operation*> op_map;

    for (auto& node : graph.nodes()) {
        std::vector<Operation*> input_ops;
        for (auto& inp : node->inputs())
            input_ops.push_back(op_map.at(inp.get()));

        auto op = make_operation(node.get(), input_ops);
        op_map[node.get()] = op.get();

        operations_.push_back(std::move(op));
    }
}

void Network::cache_data() {
    std::unordered_set<op::Input*> seen_inputs;
    std::unordered_set<Param*> seen_params;
    std::unordered_set<SelectIndices*> seen_select_indices;

    for (auto& op : operations_) {
        if (auto* p = op->param())
            if (seen_params.insert(p).second)
                params_.push_back(p);
        if (auto* sel = dynamic_cast<op::Select*>(op.get()))
            if (seen_select_indices.insert(sel->indices()).second)
                select_indices_.push_back(sel->indices());
        if (auto* inp = dynamic_cast<op::Input*>(op.get()))
            if (seen_inputs.insert(inp).second)
                inputs_.push_back(inp);
    }
}

UPtr<Operation> Network::make_operation(Node* node, std::vector<Operation*> inputs) {
    switch (node->op_type()) {
    case OpType::Input:
        return std::make_unique<op::Input>(node->output_dim());
    case OpType::SparseAffine: {
        auto* sn = dynamic_cast<SparseAffineNode*>(node);
        CHECK(sn);

        auto* input = dynamic_cast<op::Input*>(inputs[0]);

        UPtr<op::SparseAffineBase> op;
        if (sn->is_pairwise_fused())
            op = std::make_unique<op::SparseAffinePairwiseMul>(sn->param(), input);
        else
            op = std::make_unique<op::SparseAffine>(sn->param(), input);

        auto act = sn->activation();
        if (act != OpType::None)
            op->set_activation(act);

        return op;
    }
    case OpType::Affine: {
        auto* an = dynamic_cast<AffineNode*>(node);
        CHECK(an);
        return std::make_unique<op::Affine>(an->param(), inputs[0]);
    }
    case OpType::Concat: {
        auto* cn = dynamic_cast<ConcatNode*>(node);
        CHECK(cn);

        UPtr<op::ConcatBase> op;
        if (cn->is_fused()) {
            op = std::make_unique<op::FusedConcat>(inputs);
            for (auto* in : inputs) {
                if (auto* sa = dynamic_cast<op::SparseAffineBase*>(in))
                    sa->fuse_with_concat(dynamic_cast<op::FusedConcat*>(op.get()));
                else
                    CHECK(false);
            }
        } else {
            op = std::make_unique<op::Concat>(inputs);
        }

        return op;
    }
    case OpType::Select: {
        auto* sn = dynamic_cast<SelectNode*>(node);
        CHECK(sn);
        return std::make_unique<op::Select>(inputs[0], sn->indices());
    }
    case OpType::PairwiseMul:
        return std::make_unique<op::PairwiseMul>(inputs[0]);
    case OpType::ReLU:
        return make_unary<kernel::ReLU>(node, inputs[0]);
    case OpType::ClippedReLU:
        return make_unary<kernel::ClippedReLU>(node, inputs[0]);
    case OpType::SqrClippedReLU:
        return make_unary<kernel::SqrClippedReLU>(node, inputs[0]);
    case OpType::Sigmoid:
        return make_unary<kernel::Sigmoid>(node, inputs[0]);
    case OpType::AddUnary:
        return make_unary<kernel::AddUnary>(node, inputs[0]);
    case OpType::SubUnary:
        return make_unary<kernel::SubUnary>(node, inputs[0]);
    case OpType::MulUnary:
        return make_unary<kernel::MulUnary>(node, inputs[0]);
    case OpType::DivUnary:
        return make_unary<kernel::DivUnary>(node, inputs[0]);
    case OpType::AddBinary:
        return make_binary<kernel::AddBinary>(node, inputs[0], inputs[1]);
    case OpType::SubBinary:
        return make_binary<kernel::SubBinary>(node, inputs[0], inputs[1]);
    case OpType::MulBinary:
        return make_binary<kernel::MulBinary>(node, inputs[0], inputs[1]);
    case OpType::DivBinary:
        return make_binary<kernel::DivBinary>(node, inputs[0], inputs[1]);
    default:
        CHECK(false);
        return nullptr;
    }
}

} // namespace nn
