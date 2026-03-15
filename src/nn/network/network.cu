#include "network.h"

namespace nn {

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
    auto maybe_set_act = [&](auto* op, OpType act) {
        if (act != OpType::None)
            op->set_activation(to_activation(act));
    };

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

        maybe_set_act(op.get(), sn->activation());
        return op;
    }
    case OpType::Affine: {
        auto* an = dynamic_cast<AffineNode*>(node);
        CHECK(an);

        auto op = std::make_unique<op::Affine>(an->param(), inputs[0]);
        maybe_set_act(op.get(), an->activation());
        return op;
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
            maybe_set_act(op.get(), cn->activation());
        }

        return op;
    }
    case OpType::Select: {
        auto* sn = dynamic_cast<SelectNode*>(node);
        CHECK(sn);

        auto op = std::make_unique<op::Select>(inputs[0], sn->indices());
        maybe_set_act(op.get(), sn->activation());
        return op;
    }
    case OpType::PairwiseMul: {
        auto* pmn = dynamic_cast<PairwiseMulNode*>(node);
        CHECK(pmn);

        auto op = std::make_unique<op::PairwiseMul>(inputs[0]);
        maybe_set_act(op.get(), pmn->activation());
        return op;
    }
    case OpType::ReLU:
    case OpType::ClippedReLU:
    case OpType::SqrClippedReLU:
    case OpType::Sigmoid:
        return std::make_unique<op::Activation>(inputs[0], to_activation(node->op_type()));
    case OpType::Add:
        return std::make_unique<op::Elemwise<kernel::Add>>(inputs[0], inputs[1]);
    case OpType::Sub:
        return std::make_unique<op::Elemwise<kernel::Sub>>(inputs[0], inputs[1]);
    case OpType::Mul:
        return std::make_unique<op::Elemwise<kernel::Mul>>(inputs[0], inputs[1]);
    case OpType::Div:
        return std::make_unique<op::Elemwise<kernel::Div>>(inputs[0], inputs[1]);
    default:
        CHECK(false);
        return nullptr;
    }
}

} // namespace nn
