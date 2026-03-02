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

        init_select_index_fns(graph);
        init_operations(graph);
    }

    ~Network() { kernel::destroy_cublas(); }

    void init(int batch_size) {
        if (operations.size() == 0)
            error("Network: No operations found!");

        for (auto& op : operations)
            op->init(batch_size);
        for (auto& indices : select_index_fns)
            indices->init(batch_size);
    }

    void forward(const std::vector<TrainingDataEntry>& data_entries) {
        for (size_t i = 0; i < operations.size(); i++)
            operations[i]->clear_grads();
        for (auto& indices : select_index_fns)
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

    std::vector<op::Input*> get_inputs() {
        std::vector<op::Input*> result;
        for (auto& op : operations)
            if (auto* input = dpc<op::Input>(op.get()))
                result.push_back(input);
        return result;
    }

    std::vector<Param*> get_params() {
        std::vector<Param*> params;
        std::unordered_set<Param*> seen;

        for (auto& l : operations) {
            auto* m = l->get_param();
            if (m && seen.insert(m).second)
                params.push_back(m);
        }

        return params;
    }

  private:
    std::vector<Ptr<Operation>> operations;
    std::vector<SelectIndices*> select_index_fns;

    void init_select_index_fns(const Graph& graph) {
        std::unordered_set<SelectIndices*> seen;

        for (const auto& node : graph.get_nodes()) {
            if (auto* select_node = dpc<SelectNode>(node.get())) {
                auto* indices = select_node->get_indices();
                if (seen.insert(indices).second)
                    select_index_fns.push_back(indices);
            }
        }
    }

    void init_operations(const Graph& graph) {
        auto& nodes = graph.get_nodes();
        std::unordered_map<Node*, Operation*> op_map;

        for (auto& node : nodes) {
            std::vector<Operation*> input_ops;
            for (auto* input : node->get_inputs())
                input_ops.push_back(op_map.at(input));

            auto op = make_operation(node.get(), input_ops);
            op_map[node.get()] = op.get();
            operations.push_back(std::move(op));
        }
    }

    Ptr<Operation> make_operation(Node* node, std::vector<Operation*> inputs) {
        auto set_activation_if_any = [&](auto* op, OpType act) {
            if (act != OpType::None)
                op->set_activation(node_op_to_activation(act));
        };

        switch (node->get_op_type()) {
        case OpType::Input:
            return make_ptr<op::Input>(node->get_output_dim());
        case OpType::SparseAffine: {
            auto* sn = dpc<SparseAffineNode>(node);
            CHECK(sn);

            auto op = make_ptr<op::SparseAffine>(sn->get_param(), dpc<Input>(inputs[0]));
            set_activation_if_any(op.get(), sn->get_activation());

            if (sn->is_pairwise_fused())
                op->set_pairwise_fused();

            return op;
        }
        case OpType::Affine: {
            auto* an = dpc<AffineNode>(node);
            CHECK(an);

            auto op = make_ptr<op::Affine>(an->get_param(), inputs[0]);
            set_activation_if_any(op.get(), an->get_activation());
            return op;
        }
        case OpType::Concat: {
            auto* cn = dpc<ConcatNode>(node);
            CHECK(cn);

            auto op = make_ptr<op::Concat>(inputs);

            if (cn->is_fused()) {
                op->set_skip();
                for (auto* in : inputs) {
                    if (auto* sa = dpc<op::SparseAffine>(in))
                        sa->set_concat(op.get());
                    else if (auto* pm = dpc<op::PairwiseMul>(in))
                        pm->set_concat(op.get());
                    else
                        CHECK(false);
                }
            } else {
                set_activation_if_any(op.get(), cn->get_activation());
            }

            return op;
        }
        case OpType::Select: {
            auto* sn = dpc<SelectNode>(node);
            CHECK(sn);

            auto op = make_ptr<op::Select>(inputs[0], sn->get_indices());
            set_activation_if_any(op.get(), sn->get_activation());
            return op;
        }
        case OpType::PairwiseMul: {
            auto* pmn = dpc<PairwiseMulNode>(node);
            CHECK(pmn);

            auto op = make_ptr<op::PairwiseMul>(inputs[0]);
            set_activation_if_any(op.get(), pmn->get_activation());
            return op;
        }
        case OpType::ReLU:
        case OpType::ClippedReLU:
        case OpType::SqrClippedReLU:
        case OpType::Sigmoid:
            return make_ptr<op::Activate>(inputs[0], node_op_to_activation(node->get_op_type()));
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
