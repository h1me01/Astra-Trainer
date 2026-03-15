#include "graph.h"

namespace nn::graph {

void Graph::print() const {
    std::unordered_map<Node*, int> index;
    for (int i = 0; i < (int)nodes_.size(); i++)
        index[nodes_[i].get()] = i;

    for (int i = 0; i < (int)nodes_.size(); i++) {
        auto* n = nodes_[i].get();

        std::string inputs;
        for (auto& inp : n->inputs()) {
            if (!inputs.empty())
                inputs += ", ";
            inputs += std::to_string(index[inp.get()]);
        }

        std::string extra;
        if (n->activation() != OpType::None)
            extra += " [+" + op_type_str(n->activation()) + "]";
        if (auto* sa = dynamic_cast<SparseAffineNode*>(n); sa && sa->is_pairwise_fused())
            extra += " [+PairwiseMul]";
        if (auto* cn = dynamic_cast<ConcatNode*>(n); cn && cn->is_fused())
            extra += " [Fused]";

        std::cout << "[" << std::right << std::setw(2) << i << "] " << std::left << std::setw(18) << n->op_type_as_str()
                  << " dim=" << std::setw(4) << n->output_dim() << (inputs.empty() ? "" : " <- [" + inputs + "]")
                  << extra << "\n";
    }
}

void Graph::topological_sort(SPtr<Node> output) {
    std::unordered_set<Node*> visited, in_stack;

    std::function<void(SPtr<Node>)> dfs = [&](SPtr<Node> node) {
        if (in_stack.contains(node.get()))
            error("Graph: Cycle detected!");
        if (visited.contains(node.get()))
            return;

        in_stack.insert(node.get());
        for (auto& inp : node->inputs())
            dfs(inp);
        in_stack.erase(node.get());
        visited.insert(node.get());

        nodes_.push_back(node);
    };

    dfs(output);
}

void Graph::fuse_sparse_affine() {
    fixed_point<SparseAffineNode>([this](auto node) -> bool {
        auto* sa = dynamic_cast<SparseAffineNode*>(node.get());

        auto c1 = sole_consumer(node);
        if (!c1)
            return false;

        auto c1_type = c1->op_type();

        // Pass 1: try SparseAffine + Activation
        if (is_activation(c1_type)) {
            sa->set_activation(c1_type);
            absorb_node(c1, node);

            // Pass 2: try PairwiseMul fusion on top of that
            auto c2 = sole_consumer(node);
            if (c2 && c2->op_type() == OpType::PairwiseMul) {
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

void Graph::fuse_concat() {
    // Pass 1: try fusing Concat + SparseAffine
    for (auto& node : nodes_) {
        auto* cn = dynamic_cast<ConcatNode*>(node.get());
        if (!cn || cn->is_fused())
            continue;

        bool ok = std::ranges::all_of(node->inputs(), [&](auto& inp) {
            return sole_consumer(inp) == node && inp->op_type() == OpType::SparseAffine;
        });
        if (ok)
            cn->set_fused();
    }

    // Pass 2: try fusing activation into fused-concat inputs, or into unfused concat directly
    fixed_point<ConcatNode>([this](auto node) -> bool {
        auto* cn = dynamic_cast<ConcatNode*>(node.get());
        if (cn->is_fused()) {
            auto c = sole_consumer(node);
            if (!c || !is_activation(c->op_type()))
                return false;

            bool valid = std::ranges::all_of(node->inputs(), [&](auto& in) {
                return sole_consumer(in) == node && !is_activation(in->activation());
            });

            if (!valid)
                return false;

            for (auto& in : node->inputs())
                in->set_activation(c->op_type());

            absorb_node(c, node);
            return true;
        }

        return try_fuse_activation(node);
    });
}

} // namespace nn::graph
