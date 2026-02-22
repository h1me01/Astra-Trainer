#pragma once

#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "../ops/include.h"

namespace nn {

using namespace op;

class Network {
  public:
    Network() { kernel::create_cublas(); }
    ~Network() { kernel::destroy_cublas(); }

    void set_output(SPtr<Operation> op) {
        if (!op)
            error("Output operation cannot be null!");
        operations.clear();

        std::unordered_map<Operation*, SPtr<Operation>> all_ops;
        std::unordered_map<Operation*, int> in_degree;

        std::function<void(SPtr<Operation>)> collect = [&](SPtr<Operation> node) {
            if (all_ops.count(node.get()))
                return;
            all_ops[node.get()] = node;
            in_degree[node.get()] = node->get_inputs().size();
            for (auto& input : node->get_inputs())
                collect(input);
        };
        collect(op);

        std::unordered_map<Operation*, std::vector<Operation*>> dependents;
        for (auto& [ptr, node] : all_ops)
            for (auto& input : node->get_inputs())
                dependents[input.get()].push_back(ptr);

        std::queue<Operation*> ready;
        for (auto& [ptr, degree] : in_degree)
            if (degree == 0)
                ready.push(ptr);

        while (!ready.empty()) {
            auto* node = ready.front();
            ready.pop();
            operations.push_back(all_ops[node]);
            for (auto* dependent : dependents[node]) {
                if (--in_degree[dependent] == 0)
                    ready.push(dependent);
            }
        }

        if (operations.size() != all_ops.size())
            error("Cycle detected in the network!");
    }

    void init(int batch_size) {
        if (operations.size() == 0)
            error("Network has no operations!");

        std::unordered_set<SelectIndices*> seen;
        for (auto& op : operations) {
            op->init(batch_size);
            if (auto select_op = dpc<Select>(op)) {
                auto indices = select_op->get_indices();
                if (seen.insert(indices.get()).second)
                    select_indices.push_back(indices);
            }
        }

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

    std::vector<SPtr<Param>> get_params() {
        std::vector<SPtr<Param>> main_params;
        std::unordered_set<Param*> seen;

        for (auto& l : operations) {
            auto m = l->get_param();
            if (m && seen.insert(m.get()).second)
                main_params.push_back(m);
        }

        return main_params;
    }

  private:
    std::vector<SPtr<SelectIndices>> select_indices;
    std::vector<SPtr<Operation>> operations;
};

} // namespace nn
