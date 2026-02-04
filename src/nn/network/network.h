#pragma once

#include <unordered_set>

#include "../ops/include.h"

namespace nn {

class Network {
  public:
    Network() {
        kernel::create_cublas();
    }

    ~Network() {
        kernel::destroy_cublas();
    }

    void add_operation(Ptr<Operation> op) {
        operations.push_back(op);
    }

    void init(int batch_size) {
        if (operations.size() == 0)
            error("Network has no operations!");
        for (auto& op : operations)
            op->init(batch_size);
    }

    void forward(const std::vector<TrainingDataEntry>& data_entries) {
        for (auto& l : operations)
            l->step(data_entries);

        for (size_t i = 0; i < operations.size(); i++)
            operations[i]->forward();
    }

    void backward() {
        for (int i = operations.size() - 1; i >= 0; i--)
            operations[i]->backward();
    }

    OpTensor& get_output() {
        return operations.back()->get_tensor_output();
    }

    const OpTensor& get_output() const {
        return operations.back()->get_tensor_output();
    }

    std::vector<Ptr<Params>> get_params() {
        std::vector<Ptr<Params>> main_params;
        std::unordered_set<Params*> seen;

        for (auto& l : operations) {
            auto m = l->get_params();
            if (m && seen.insert(m.get()).second)
                main_params.push_back(m);
        }

        return main_params;
    }

  private:
    std::vector<Ptr<Operation>> operations;
};

} // namespace nn
