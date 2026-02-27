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

        init_select_indices(graph);
        init_operations(graph);
    }

    ~Network() { kernel::destroy_cublas(); }

    void init(int batch_size) {
        if (operations.size() == 0)
            error("Network has no operations!");

        for (auto& op : operations)
            op->init(batch_size);
        for (auto& indices : select_indices)
            indices->init(batch_size);
    }

    void forward(const std::vector<TrainingDataEntry>& data_entries) {
        for (size_t i = 0; i < operations.size(); i++)
            operations[i]->clear_grads();
        for (auto& indices : select_indices)
            indices->step(data_entries);
        for (size_t i = 0; i < operations.size(); i++)
            operations[i]->forward();
    }

    void backward() {
        for (int i = operations.size() - 1; i >= 0; i--)
            operations[i]->backward();
    }

    Tensor& get_output() { return operations.back()->get_output(); }
    const Tensor& get_output() const { return operations.back()->get_output(); }

    std::vector<SPtr<op::Input>> get_inputs() {
        std::vector<SPtr<op::Input>> result;
        for (auto& op : operations)
            if (auto input = dpc<op::Input>(op))
                result.push_back(input);
        return result;
    }

    std::vector<SPtr<Param>> get_params() {
        std::vector<SPtr<Param>> params;
        std::unordered_set<Param*> seen;

        for (auto& l : operations) {
            auto m = l->get_param();
            if (m && seen.insert(m.get()).second)
                params.push_back(m);
        }

        return params;
    }

  private:
    std::vector<SPtr<Input>> inputs;
    std::vector<SPtr<Operation>> operations;
    std::vector<SPtr<SelectIndices>> select_indices;

    void init_select_indices(const Graph& graph) {
        std::unordered_set<SelectIndices*> seen;

        for (const auto& node : graph.get_nodes()) {
            if (auto select_node = dpc<SelectNode>(node)) {
                auto indices = select_node->get_indices();
                if (seen.insert(indices.get()).second)
                    select_indices.push_back(indices);
            }
        }
    }

    void init_operations(const Graph& graph) {
        auto& nodes = graph.get_nodes();
        std::unordered_map<Node*, SPtr<Operation>> op_map;

        for (auto& node : nodes) {
            std::vector<SPtr<Operation>> input_ops;
            for (auto& input : node->get_inputs())
                input_ops.push_back(op_map.at(input.get()));

            SPtr<Operation> op = make_operation(node, input_ops);
            op_map[node.get()] = op;
            operations.push_back(op);
        }
    }

    SPtr<Operation> make_operation(SPtr<Node> node, std::vector<SPtr<Operation>> inputs) {
        auto set_activation_if_any = [&](auto op, OpType act) {
            if (act != OpType::None)
                op->set_activation(node_op_to_activation(act));
        };

        switch (node->get_op_type()) {
        case OpType::Input:
            return std::make_shared<op::Input>(node->get_output_dim());
        case OpType::SparseAffine: {
            auto sn = dpc<SparseAffineNode>(node);
            CHECK(sn);

            auto op = std::make_shared<op::SparseAffine>(sn->get_param(), dpc<Input>(inputs[0]));
            set_activation_if_any(op, sn->get_activation());

            if (sn->is_pairwise_fused())
                op->set_pairwise_fused();

            return op;
        }
        case OpType::Affine: {
            auto an = dpc<AffineNode>(node);
            CHECK(an);

            auto op = std::make_shared<op::Affine>(an->get_param(), inputs[0]);
            set_activation_if_any(op, an->get_activation());
            return op;
        }
        case OpType::Concat: {
            auto cn = dpc<ConcatNode>(node);
            CHECK(cn);

            auto op = std::make_shared<op::Concat>(inputs);

            if (cn->is_fused()) {
                op->set_skip();
                for (auto& in : inputs) {
                    if (auto sa = dpc<op::SparseAffine>(in))
                        sa->set_concat(op);
                    else if (auto pm = dpc<op::PairwiseMul>(in))
                        pm->set_concat(op);
                    else
                        CHECK(false);
                }
            } else {
                set_activation_if_any(op, cn->get_activation());
            }

            return op;
        }
        case OpType::Select: {
            auto sn = dpc<SelectNode>(node);
            CHECK(sn);

            auto op = std::make_shared<op::Select>(inputs[0], sn->get_indices());
            set_activation_if_any(op, sn->get_activation());
            return op;
        }
        case OpType::PairwiseMul: {
            auto pmn = dpc<PairwiseMulNode>(node);
            CHECK(pmn);

            auto op = std::make_shared<op::PairwiseMul>(inputs[0]);
            set_activation_if_any(op, pmn->get_activation());
            return op;
        }
        case OpType::ReLU:
        case OpType::ClippedReLU:
        case OpType::SqrClippedReLU:
        case OpType::Sigmoid:
            return std::make_shared<op::Activate>(inputs[0], node_op_to_activation(node->get_op_type()));
        default:
            CHECK(false);
            return nullptr;
        }
    }

    ActivationType node_op_to_activation(OpType op_type) const {
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
};

} // namespace nn
