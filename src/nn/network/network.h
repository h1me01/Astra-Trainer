#pragma once

#include <unordered_set>

#include "../ops/include.h"
#include "../optimizer/optimizer.h"

namespace nn {

using namespace op;
using namespace optim;

class Network {
  public:
    Network() { kernel::create_cublas(); }
    ~Network() { kernel::destroy_cublas(); }

    void set_output(SPtr<Operation> op) {
        if (!op)
            error("Output operation cannot be null!");

        operations.clear();
        operations.push_back(op);
        init_operations(op->get_inputs());
    }

    void init(int batch_size) {
        if (operations.size() == 0)
            error("Network has no operations!");

        // set output will initialize operations in reverse order, so reverse it
        std::reverse(operations.begin(), operations.end());

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

    void clear_all_grads(Optimizer* optim) {
        for (size_t i = 0; i < operations.size(); i++)
            operations[i]->clear_grads();
        if (optim)
            optim->clear_grads();
    }

    void forward(const std::vector<TrainingDataEntry>& data_entries) {
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

    void init_operations(const std::vector<SPtr<Operation>>& ops) {
        if (ops.empty())
            return;

        for (const auto& l : ops) {
            operations.push_back(l);
            init_operations(l->get_inputs());
        }
    }
};

} // namespace nn
