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

    void print() const;

    const std::vector<SPtr<Node>>& nodes() const { return nodes_; }
    const std::vector<SPtr<Param>>& params() const { return params_; }
    const std::vector<SPtr<SelectIndices>>& select_indices() const { return select_indices_; }

  private:
    std::vector<SPtr<Node>> nodes_;
    std::vector<SPtr<Param>> params_;
    std::vector<SPtr<SelectIndices>> select_indices_;
    std::unordered_map<Node*, std::vector<SPtr<Node>>> consumers_;

    // Build

    void topological_sort(SPtr<Node> output);

    template <typename NodeT, typename ItemT, typename Getter>
    void collect_unique(std::vector<SPtr<ItemT>>& out, Getter get) {
        std::unordered_set<ItemT*> seen;
        for (auto& node : nodes_)
            if (auto* n = dynamic_cast<NodeT*>(node.get()))
                if (auto item = get(n); seen.insert(item.get()).second)
                    out.push_back(item);
    }

    void init_params() {
        collect_unique<SparseAffineNode, Param>(params_, [](auto* n) { return n->param(); });
        collect_unique<AffineNode, Param>(params_, [](auto* n) { return n->param(); });
    }

    void init_select_indices() {
        collect_unique<SelectNode, SelectIndices>(select_indices_, [](auto* n) { return n->indices(); });
    }

    void build_consumer_map() {
        consumers_.clear();
        for (auto& node : nodes_) {
            consumers_.emplace(node.get(), std::vector<SPtr<Node>>{});
            for (auto& inp : node->inputs())
                consumers_[inp.get()].push_back(node);
        }
    }

    // Helpers

    SPtr<Node> sole_consumer(SPtr<Node> node) const {
        auto it = consumers_.find(node.get());
        return (it != consumers_.end() && it->second.size() == 1) ? it->second[0] : nullptr;
    }

    void absorb_node(SPtr<Node> consumed, SPtr<Node> owner) {
        for (auto& node : nodes_)
            if (node != consumed)
                for (auto& inp : node->inputs())
                    if (inp == consumed)
                        inp = owner;

        std::erase_if(nodes_, [&](const auto& n) { return n == consumed; });

        build_consumer_map();
    }

    bool try_fuse_activation(SPtr<Node> node) {
        auto c = sole_consumer(node);
        if (!c || !is_activation(c->op_type()))
            return false;
        node->set_activation(c->op_type());
        absorb_node(c, node);
        return true;
    }

    template <typename T>
    void fixed_point(std::function<bool(SPtr<Node>)> action) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto& node : nodes_) {
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

    void fuse_sparse_affine();
    void fuse_concat();
};

} // namespace nn::graph
