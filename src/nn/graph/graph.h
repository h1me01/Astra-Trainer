#pragma once

#include <functional>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "node.h"

namespace nn::graph {

class Graph {
  public:
    Graph(SPtr<Node> output_node) {
        topological_sort(output_node);
        build_consumer_map();
        optimize();
    }

    void print() const {
        std::unordered_map<Node*, int> index;
        for (int i = 0; i < (int)nodes.size(); i++)
            index[nodes[i].get()] = i;

        for (int i = 0; i < (int)nodes.size(); i++) {
            auto& node = nodes[i];

            std::string inputs_str;
            for (auto& input : node->get_inputs()) {
                if (!inputs_str.empty())
                    inputs_str += ", ";
                inputs_str += std::to_string(index[input.get()]);
            }

            std::string extra;
            if (auto cn = dpc<ConcatNode>(node); cn && cn->is_fused())
                extra += " [concat-fused]";
            if (node->get_activation() != OpType::None)
                extra += " [+" + op_type_str(node->get_activation()) + "]";
            if (auto sa = dpc<SparseAffineNode>(node); sa && sa->is_pairwise_fused())
                extra += " [+PairwiseMul]";

            std::cout << "[" << std::right << std::setw(2) << i << "] "        //
                      << std::left << std::setw(18) << node->get_op_type_str() //
                      << " dim=" << std::setw(4) << node->get_output_dim()     //
                      << (inputs_str.empty() ? "" : " <- [" + inputs_str + "]") << extra << "\n";
        }
    }

    const std::vector<SPtr<Node>>& get_nodes() const { return nodes; }

  private:
    std::vector<SPtr<Node>> nodes;
    std::unordered_map<Node*, std::vector<SPtr<Node>>> consumers;

    // Build

    void topological_sort(SPtr<Node> output) {
        std::unordered_set<Node*> visited;
        std::unordered_set<Node*> in_stack;

        std::function<void(SPtr<Node>)> dfs = [&](SPtr<Node> node) {
            if (in_stack.count(node.get()))
                error("Graph: Cycle detected!");
            if (visited.count(node.get()))
                return;

            in_stack.insert(node.get());
            for (auto& input : node->get_inputs())
                dfs(input);
            in_stack.erase(node.get());

            visited.insert(node.get());
            nodes.push_back(node);
        };

        dfs(output);
    }

    void build_consumer_map() {
        consumers.clear();
        for (auto& node : nodes) {
            consumers.emplace(node.get(), std::vector<SPtr<Node>>{});
            for (auto& input : node->get_inputs())
                consumers[input.get()].push_back(node);
        }
    }

    // Helpers

    SPtr<Node> sole_consumer(SPtr<Node> node) const {
        auto it = consumers.find(node.get());
        if (it == consumers.end() || it->second.size() != 1)
            return nullptr;
        return it->second[0];
    }

    void absorb_node(SPtr<Node> consumed, SPtr<Node> owner) {
        for (auto& node : nodes) {
            if (node == consumed)
                continue;
            for (auto& inp : node->get_inputs())
                if (inp == consumed)
                    inp = owner;
        }
        nodes.erase(std::remove(nodes.begin(), nodes.end(), consumed), nodes.end());
        build_consumer_map();
    }

    bool try_fuse_activation(SPtr<Node> node) {
        SPtr<Node> c = sole_consumer(node);
        if (!c || !is_activation(c->get_op_type()))
            return false;
        node->set_activation(c->get_op_type());
        absorb_node(c, node);
        return true;
    }

    // Fusion passes

    void optimize() {
        fuse_sparse_affine();
        fuse_affine();
        fuse_select();
        fuse_pairwise_mul();
        fuse_concat();
    }

    void fuse_sparse_affine() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto& node : nodes) {
                auto sa = dpc<SparseAffineNode>(node);
                if (!sa)
                    continue;

                SPtr<Node> c1 = sole_consumer(node);
                if (!c1)
                    continue;

                if (is_activation(c1->get_op_type())) {
                    SPtr<Node> c2 = sole_consumer(c1);
                    if (c2 && c2->get_op_type() == OpType::PairwiseMul) {
                        // try activation + pairwise mul fusion first
                        sa->set_activation(c1->get_op_type());
                        sa->set_pairwise_fused();
                        absorb_node(c1, node);
                        absorb_node(c2, node);
                    } else {
                        // try only activation fusion
                        sa->set_activation(c1->get_op_type());
                        absorb_node(c1, node);
                    }
                    changed = true;
                    break;
                }

                // if no activation provided, try pairwise mul fusion
                if (c1->get_op_type() == OpType::PairwiseMul) {
                    sa->set_pairwise_fused();
                    absorb_node(c1, node);
                    changed = true;
                    break;
                }
            }
        }
    }

    void fuse_affine() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto& node : nodes) {
                if (!dpc<AffineNode>(node))
                    continue;
                if (try_fuse_activation(node)) {
                    changed = true;
                    break;
                }
            }
        }
    }

    void fuse_select() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto& node : nodes) {
                if (!dpc<SelectNode>(node))
                    continue;
                if (try_fuse_activation(node)) {
                    changed = true;
                    break;
                }
            }
        }
    }

    void fuse_pairwise_mul() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto& node : nodes) {
                if (!dpc<PairwiseMulNode>(node))
                    continue;
                if (try_fuse_activation(node)) {
                    changed = true;
                    break;
                }
            }
        }
    }

    void fuse_concat() {
        // try fusing with sparse affine / pairwise mul first
        for (auto& node : nodes) {
            auto cn = dpc<ConcatNode>(node);
            if (!cn || cn->is_fused())
                continue;

            bool all_fusable = true;
            for (auto& input : node->get_inputs()) {
                if (sole_consumer(input).get() != node.get()) {
                    all_fusable = false;
                    break;
                }

                OpType t = input->get_op_type();
                if (t != OpType::SparseAffine && t != OpType::PairwiseMul) {
                    all_fusable = false;
                    break;
                }
            }

            if (all_fusable)
                cn->set_fused();
        }

        // try fusing with activation
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto& node : nodes) {
                auto cn = dpc<ConcatNode>(node);
                if (!cn || cn->is_fused())
                    continue;
                if (try_fuse_activation(node)) {
                    changed = true;
                    break;
                }
            }
        }
    }
};

} // namespace nn::graph
