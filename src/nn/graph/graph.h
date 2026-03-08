#pragma once

#include <functional>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "../../misc.h"
#include "node.h"

namespace nn::graph {

class Graph {
  public:
    Graph(SPtr<Node> output) {
        topological_sort(output);
        init_params();
        init_select_indices();
        build_consumer_map();
        optimize();
    }

    void print() const {
        std::unordered_map<Node*, int> index;
        for (int i = 0; i < (int)nodes.size(); i++)
            index[nodes[i].get()] = i;

        for (int i = 0; i < (int)nodes.size(); i++) {
            auto* n = nodes[i].get();

            std::string inputs;
            for (auto& inp : n->get_inputs()) {
                if (!inputs.empty())
                    inputs += ", ";
                inputs += std::to_string(index[inp.get()]);
            }

            std::string extra;
            if (n->get_activation() != OpType::None)
                extra += " [+" + op_type_str(n->get_activation()) + "]";
            if (auto* sa = dynamic_cast<SparseAffineNode*>(n); sa && sa->is_pairwise_fused())
                extra += " [+PairwiseMul]";
            if (auto* cn = dynamic_cast<ConcatNode*>(n); cn && cn->is_fused())
                extra += " [Fused]";

            std::cout << "[" << std::right << std::setw(2) << i << "] " << std::left << std::setw(18)
                      << n->get_op_type_str() << " dim=" << std::setw(4) << n->get_output_dim()
                      << (inputs.empty() ? "" : " <- [" + inputs + "]") << extra << "\n";
        }
    }

    const std::vector<SPtr<Node>>& get_nodes() const { return nodes; }
    const std::vector<SPtr<Param>>& get_params() const { return params; }
    const std::vector<SPtr<SelectIndices>>& get_select_indices() const { return select_indices; }

  private:
    std::vector<SPtr<Node>> nodes;
    std::vector<SPtr<Param>> params;
    std::vector<SPtr<SelectIndices>> select_indices;
    std::unordered_map<Node*, std::vector<SPtr<Node>>> consumers;

    // Build

    void topological_sort(SPtr<Node> output) {
        std::unordered_set<Node*> visited, in_stack;

        std::function<void(SPtr<Node>)> dfs = [&](SPtr<Node> node) {
            if (in_stack.contains(node.get()))
                error("Graph: Cycle detected!");
            if (visited.contains(node.get()))
                return;

            in_stack.insert(node.get());
            for (auto& inp : node->get_inputs())
                dfs(inp);
            in_stack.erase(node.get());
            visited.insert(node.get());

            nodes.push_back(node);
        };

        dfs(output);
    }

    template <typename NodeT, typename ItemT, typename Getter>
    void collect_unique(std::vector<SPtr<ItemT>>& out, Getter get) {
        std::unordered_set<ItemT*> seen;
        for (auto& node : nodes)
            if (auto* n = dynamic_cast<NodeT*>(node.get()))
                if (auto item = get(n); seen.insert(item.get()).second)
                    out.push_back(item);
    }

    void init_params() {
        collect_unique<SparseAffineNode, Param>(params, [](auto* n) { return n->get_param(); });
        collect_unique<AffineNode, Param>(params, [](auto* n) { return n->get_param(); });
    }

    void init_select_indices() {
        collect_unique<SelectNode, SelectIndices>(select_indices, [](auto* n) { return n->get_indices(); });
    }

    void build_consumer_map() {
        consumers.clear();
        for (auto& node : nodes) {
            consumers.emplace(node.get(), std::vector<SPtr<Node>>{});
            for (auto& inp : node->get_inputs())
                consumers[inp.get()].push_back(node);
        }
    }

    // Helpers

    SPtr<Node> sole_consumer(SPtr<Node> node) const {
        auto it = consumers.find(node.get());
        return (it != consumers.end() && it->second.size() == 1) ? it->second[0] : nullptr;
    }

    void absorb_node(SPtr<Node> consumed, SPtr<Node> owner) {
        for (auto& node : nodes)
            if (node != consumed)
                for (auto& inp : node->get_inputs())
                    if (inp == consumed)
                        inp = owner;

        std::erase_if(nodes, [&](const auto& n) { return n == consumed; });

        build_consumer_map();
    }

    bool try_fuse_activation(SPtr<Node> node) {
        auto c = sole_consumer(node);
        if (!c || !is_activation(c->get_op_type()))
            return false;
        node->set_activation(c->get_op_type());
        absorb_node(c, node);
        return true;
    }

    template <typename T>
    void fixed_point(std::function<bool(SPtr<Node>)> action) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto& node : nodes) {
                if (!dynamic_cast<T*>(node.get()))
                    continue;
                if (action(node)) {
                    changed = true;
                    break;
                }
            }
        }
    }

    // Fusion passes

    void optimize() {
        fuse_sparse_affine();
        fixed_point<AffineNode>([this](auto n) { return try_fuse_activation(n); });
        fixed_point<SelectNode>([this](auto n) { return try_fuse_activation(n); });
        fixed_point<PairwiseMulNode>([this](auto n) { return try_fuse_activation(n); });
        fuse_concat();
    }

    void fuse_sparse_affine() {
        fixed_point<SparseAffineNode>([this](auto node) -> bool {
            auto* sa = dynamic_cast<SparseAffineNode*>(node.get());

            auto c1 = sole_consumer(node);
            if (!c1)
                return false;

            auto c1_type = c1->get_op_type();

            // Pass 1: try SparseAffine + Activation
            if (is_activation(c1_type)) {
                sa->set_activation(c1_type);
                absorb_node(c1, node);

                // Pass 2: try PairwiseMul fusion on top of that
                auto c2 = sole_consumer(node);
                if (c2 && c2->get_op_type() == OpType::PairwiseMul) {
                    sa->set_pairwise_fused();
                    absorb_node(c2, node);
                }

                return true;
            }

            // Pass 2: try SparseAffine + PairwiseMul
            if (c1_type == OpType::PairwiseMul) {
                sa->set_pairwise_fused();
                absorb_node(c1, node);
                return true;
            }

            return false;
        });
    }

    void fuse_concat() {
        // Pass 1: try fusing Concat + SparseAffine
        for (auto& node : nodes) {
            auto* cn = dynamic_cast<ConcatNode*>(node.get());
            if (!cn || cn->is_fused())
                continue;

            bool ok = std::ranges::all_of(node->get_inputs(), [&](auto& inp) {
                return sole_consumer(inp) == node && inp->get_op_type() == OpType::SparseAffine;
            });
            if (ok)
                cn->set_fused();
        }

        // Pass 2: try fusing activation into fused-concat inputs, or into unfused concat directly
        fixed_point<ConcatNode>([this](auto node) -> bool {
            auto* cn = dynamic_cast<ConcatNode*>(node.get());
            if (cn->is_fused()) {
                auto c = sole_consumer(node);
                if (!c || !is_activation(c->get_op_type()))
                    return false;

                bool valid = std::ranges::all_of(node->get_inputs(), [&](auto& in) {
                    return sole_consumer(in) == node && !is_activation(in->get_activation());
                });

                if (!valid)
                    return false;

                for (auto& in : node->get_inputs())
                    in->set_activation(c->get_op_type());

                absorb_node(c, node);
                return true;
            }

            return try_fuse_activation(node);
        });
    }
};

} // namespace nn::graph
