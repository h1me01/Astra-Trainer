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
    struct BuildContext {
        std::vector<Ptr<Node>> nodes;
        std::vector<Ptr<Param>> params;
        std::vector<Ptr<SelectIndices>> select_indices;

        inline static thread_local BuildContext* active = nullptr;

        static Node* register_node(Ptr<Node> n) {
            CHECK(active);
            auto* ptr = n.get();
            active->nodes.push_back(std::move(n));
            return ptr;
        }

        static Param* register_param(Ptr<Param> p) {
            CHECK(active);
            auto* ptr = p.get();
            active->params.push_back(std::move(p));
            return ptr;
        }

        static SelectIndices* register_select_indices(Ptr<SelectIndices> si) {
            CHECK(active);
            auto* ptr = si.get();
            active->select_indices.push_back(std::move(si));
            return ptr;
        }
    };

    Graph(Node* output, BuildContext ctx) {
        params = std::move(ctx.params);
        select_indices = std::move(ctx.select_indices);

        std::vector<Node*> ordered;
        topological_sort(output, ordered);

        std::unordered_map<Node*, Ptr<Node>> node_map;
        for (auto& n : ctx.nodes)
            node_map[n.get()] = std::move(n);
        for (Node* ptr : ordered)
            nodes.push_back(std::move(node_map.at(ptr)));

        build_consumer_map();
        optimize();
    }

    void print() const {
        std::unordered_map<Node*, int> index;
        for (int i = 0; i < (int)nodes.size(); i++)
            index[nodes[i].get()] = i;

        for (int i = 0; i < (int)nodes.size(); i++) {
            auto* node = nodes[i].get();

            std::string inputs_str;
            for (auto* input : node->get_inputs()) {
                if (!inputs_str.empty())
                    inputs_str += ", ";
                inputs_str += std::to_string(index[input]);
            }

            std::string extra;
            if (auto* cn = dpc<ConcatNode>(node); cn && cn->is_fused())
                extra += " [concat-fused]";
            if (node->get_activation() != OpType::None)
                extra += " [+" + op_type_str(node->get_activation()) + "]";
            if (auto* sa = dpc<SparseAffineNode>(node); sa && sa->is_pairwise_fused())
                extra += " [+PairwiseMul]";

            std::cout << "[" << std::right << std::setw(2) << i << "] "        //
                      << std::left << std::setw(18) << node->get_op_type_str() //
                      << " dim=" << std::setw(4) << node->get_output_dim()     //
                      << (inputs_str.empty() ? "" : " <- [" + inputs_str + "]") << extra << "\n";
        }
    }

    const std::vector<Ptr<Node>>& get_nodes() const { return nodes; }
    const std::vector<Ptr<SelectIndices>>& get_select_indices() const { return select_indices; }

  private:
    std::vector<Ptr<Node>> nodes;
    std::vector<Ptr<Param>> params;
    std::vector<Ptr<SelectIndices>> select_indices;
    std::unordered_map<Node*, std::vector<Node*>> consumers;

    // Build

    void topological_sort(Node* output, std::vector<Node*>& ordered) {
        std::unordered_set<Node*> visited;
        std::unordered_set<Node*> in_stack;

        std::function<void(Node*)> dfs = [&](Node* node) {
            if (in_stack.count(node))
                error("Graph: Cycle detected!");
            if (visited.count(node))
                return;

            in_stack.insert(node);
            for (auto* input : node->get_inputs())
                dfs(input);
            in_stack.erase(node);

            visited.insert(node);
            ordered.push_back(node);
        };

        dfs(output);
    }

    void build_consumer_map() {
        consumers.clear();
        for (auto& node : nodes) {
            consumers.emplace(node.get(), std::vector<Node*>{});
            for (auto* input : node->get_inputs())
                consumers[input].push_back(node.get());
        }
    }

    // Helpers

    Node* sole_consumer(Node* node) const {
        auto it = consumers.find(node);
        if (it == consumers.end() || it->second.size() != 1)
            return nullptr;
        return it->second[0];
    }

    void absorb_node(Node* consumed, Node* owner) {
        for (auto& node : nodes) {
            if (node.get() == consumed)
                continue;
            for (auto*& inp : node->get_inputs())
                if (inp == consumed)
                    inp = owner;
        }
        nodes.erase(
            std::remove_if(nodes.begin(), nodes.end(), [consumed](const auto& n) { return n.get() == consumed; }),
            nodes.end()
        );
        build_consumer_map();
    }

    bool try_fuse_activation(Node* node) {
        Node* c = sole_consumer(node);
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
                auto* sa = dpc<SparseAffineNode>(node.get());
                if (!sa)
                    continue;

                Node* c1 = sole_consumer(node.get());
                if (!c1)
                    continue;

                if (is_activation(c1->get_op_type())) {
                    Node* c2 = sole_consumer(c1);
                    if (c2 && c2->get_op_type() == OpType::PairwiseMul) {
                        // try activation + pairwise mul fusion first
                        sa->set_activation(c1->get_op_type());
                        sa->set_pairwise_fused();
                        absorb_node(c1, node.get());
                        absorb_node(c2, node.get());
                    } else {
                        // try only activation fusion
                        sa->set_activation(c1->get_op_type());
                        absorb_node(c1, node.get());
                    }
                    changed = true;
                    break;
                }

                // if no activation provided, try pairwise mul fusion
                if (c1->get_op_type() == OpType::PairwiseMul) {
                    sa->set_pairwise_fused();
                    absorb_node(c1, node.get());
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
                if (!dpc<AffineNode>(node.get()))
                    continue;
                if (try_fuse_activation(node.get())) {
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
                if (!dpc<SelectNode>(node.get()))
                    continue;
                if (try_fuse_activation(node.get())) {
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
                if (!dpc<PairwiseMulNode>(node.get()))
                    continue;
                if (try_fuse_activation(node.get())) {
                    changed = true;
                    break;
                }
            }
        }
    }

    void fuse_concat() {
        // try fusing with sparse affine / pairwise mul first
        for (auto& node : nodes) {
            auto* cn = dpc<ConcatNode>(node.get());
            if (!cn || cn->is_fused())
                continue;

            bool all_fusable = true;
            for (auto* input : node->get_inputs()) {
                if (sole_consumer(input) != node.get()) {
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
                auto* cn = dpc<ConcatNode>(node.get());
                if (!cn || cn->is_fused())
                    continue;
                if (try_fuse_activation(node.get())) {
                    changed = true;
                    break;
                }
            }
        }
    }
};

} // namespace nn::graph
