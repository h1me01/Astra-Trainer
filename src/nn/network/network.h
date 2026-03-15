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
    Network(const SPtr<Node> output) {
        kernel::cublas::create();

        auto graph = Graph(output);

        init_operations(graph);
        cache_data();
    }

    ~Network() { kernel::cublas::destroy(); }

    void init(int batch_size) {
        if (operations_.empty())
            error("Network: No operations found!");
        for (auto& op : operations_)
            op->init(batch_size);
        for (auto& idx : select_indices_)
            idx->init(batch_size);
    }

    void forward(const std::vector<TrainingDataEntry>& data_entries) {
        for (auto& op : operations_)
            op->zero_grads();
        for (auto& idx : select_indices_)
            idx->step(data_entries);

        for (auto& p : params_)
            if (p->has_factorizer())
                p->factorizer().forward();
        for (auto& op : operations_)
            op->forward();
    }

    void backward() {
        for (int i = (int)operations_.size() - 1; i >= 0; i--)
            operations_[i]->backward();
        for (auto& p : params_)
            if (p->has_factorizer())
                p->factorizer().backward();
    }

    Tensor& output() { return operations_.back()->output(); }
    const Tensor& output() const { return operations_.back()->output(); }

    const std::vector<op::Input*>& inputs() const { return inputs_; }
    const std::vector<Param*>& params() const { return params_; }

  private:
    std::vector<UPtr<Operation>> operations_;
    std::vector<Param*> params_;
    std::vector<SelectIndices*> select_indices_;
    std::vector<op::Input*> inputs_;

    void init_operations(const Graph& graph);
    void cache_data();

    static ActivationType to_activation(OpType op_type) {
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

    UPtr<Operation> make_operation(Node* node, std::vector<Operation*> inputs);
};

} // namespace nn
