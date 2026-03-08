#pragma once

#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "../graph/graph.h"
#include "../ops/include.h"

namespace nn {

using namespace graph;
using namespace op;

class Network {
  public:
    Network(const Graph& graph) {
        kernel::create_cublas();
        init_operations(graph);
        cache_data();
    }

    ~Network() { kernel::destroy_cublas(); }

    void init(int batch_size) {
        if (operations.empty())
            error("Network: No operations found!");
        for (auto& op : operations)
            op->init(batch_size);
        for (auto& idx : select_indices)
            idx->init(batch_size);
    }

    void forward(const std::vector<TrainingDataEntry>& data_entries) {
        for (auto& op : operations)
            op->zero_grads();
        for (auto& idx : select_indices)
            idx->step(data_entries);

        for (auto& p : params)
            if (p->has_factorizer())
                p->get_factorizer().forward();
        for (auto& op : operations)
            op->forward();
    }

    void backward() {
        for (int i = (int)operations.size() - 1; i >= 0; i--)
            operations[i]->backward();
        for (auto& p : params)
            if (p->has_factorizer())
                p->get_factorizer().backward();
    }

    Tensor& get_output() { return operations.back()->get_output(); }
    const Tensor& get_output() const { return operations.back()->get_output(); }

    const std::vector<op::Input*>& get_inputs() const { return inputs; }
    const std::vector<Param*>& get_params() const { return params; }

  private:
    std::vector<UPtr<Operation>> operations;
    std::vector<Param*> params;
    std::vector<SelectIndices*> select_indices;
    std::vector<op::Input*> inputs;

    void init_operations(const Graph& graph) {
        std::unordered_map<Node*, Operation*> op_map;

        for (auto& node : graph.get_nodes()) {
            std::vector<Operation*> input_ops;
            for (auto& inp : node->get_inputs())
                input_ops.push_back(op_map.at(inp.get()));

            auto op = make_operation(node.get(), input_ops);
            op_map[node.get()] = op.get();

            operations.push_back(std::move(op));
        }
    }

    void cache_data() {
        std::unordered_set<op::Input*> seen_inputs;
        std::unordered_set<Param*> seen_params;
        std::unordered_set<SelectIndices*> seen_select_indices;

        for (auto& op : operations) {
            if (auto* p = op->get_param())
                if (seen_params.insert(p).second)
                    params.push_back(p);
            if (auto* sel = dynamic_cast<op::Select*>(op.get()))
                if (seen_select_indices.insert(sel->get_indices()).second)
                    select_indices.push_back(sel->get_indices());
            if (auto* inp = dynamic_cast<op::Input*>(op.get()))
                if (seen_inputs.insert(inp).second)
                    inputs.push_back(inp);
        }
    }

    static ActivationType to_activation(OpType op_type) {
        switch (op_type) {
        case OpType::ReLU:
            return ActivationType::ReLU;
        case OpType::ClippedReLU:
            return ActivationType::ClippedReLU;
        case OpType::SqrClippedReLU:
            return ActivationType::SqrClippedReLU;
        case OpType::Sigmoid:
            return ActivationType::Sigmoid;
        default:
            CHECK(false);
            return ActivationType::Linear;
        }
    }

    UPtr<Operation> make_operation(Node* node, std::vector<Operation*> inputs) {
        auto maybe_set_act = [&](auto* op, OpType act) {
            if (act != OpType::None)
                op->set_activation(to_activation(act));
        };

        switch (node->get_op_type()) {
        case OpType::Input:
            return std::make_unique<op::Input>(node->get_output_dim());
        case OpType::SparseAffine: {
            auto* sn = dynamic_cast<SparseAffineNode*>(node);
            CHECK(sn);

            auto* input = dynamic_cast<op::Input*>(inputs[0]);

            UPtr<op::SparseAffineBase> op;
            if (sn->is_pairwise_fused())
                op = std::make_unique<op::SparseAffinePairwiseMul>(sn->get_param(), input);
            else
                op = std::make_unique<op::SparseAffine>(sn->get_param(), input);

            maybe_set_act(op.get(), sn->get_activation());
            return op;
        }
        case OpType::Affine: {
            auto* an = dynamic_cast<AffineNode*>(node);
            CHECK(an);

            auto op = std::make_unique<op::Affine>(an->get_param(), inputs[0]);
            maybe_set_act(op.get(), an->get_activation());
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
                maybe_set_act(op.get(), cn->get_activation());
            }

            return op;
        }
        case OpType::Select: {
            auto* sn = dynamic_cast<SelectNode*>(node);
            CHECK(sn);

            auto op = std::make_unique<op::Select>(inputs[0], sn->get_indices());
            maybe_set_act(op.get(), sn->get_activation());
            return op;
        }
        case OpType::PairwiseMul: {
            auto* pmn = dynamic_cast<PairwiseMulNode*>(node);
            CHECK(pmn);

            auto op = std::make_unique<op::PairwiseMul>(inputs[0]);
            maybe_set_act(op.get(), pmn->get_activation());
            return op;
        }
        case OpType::ReLU:
        case OpType::ClippedReLU:
        case OpType::SqrClippedReLU:
        case OpType::Sigmoid:
            return std::make_unique<op::Activation>(inputs[0], to_activation(node->get_op_type()));
        default:
            CHECK(false);
            return nullptr;
        }
    }
};

} // namespace nn
