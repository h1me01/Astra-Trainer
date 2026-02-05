#pragma once

#include "ops.h"

namespace nn {

class Select : public Operation {
  public:
    Select(Ptr<Operation> input, const std::function<int(const Position&)>& select_fn)
        : input(input),
          select_fn(select_fn) {

        max_indices = select_fn(Position::startPosition()) + 1;

        input_dim = input->get_output_dim();
        output_dim = input_dim / max_indices;

        if (input_dim % max_indices != 0)
            error("Select input dimension must be a multiple of select size!");
    }

    void init(int batch_size) override {
        Operation::init(batch_size);
        indices = Array<int>(batch_size);
    }

    void step(const std::vector<TrainingDataEntry>& data_entries) override {
        Operation::step(data_entries);

        for (int i = 0; i < (int)data_entries.size(); i++) {
            int idx = select_fn(data_entries[i].pos);
            if (idx < 0 || idx >= max_indices)
                error("Index function of Select returned invalid index!");
            indices(i) = idx;
        }

        indices.host_to_dev();
    }

    void forward() override {
        kernel::select_fwd(
            input->get_output(), tensor_output.get_linear_output(), tensor_output.get_activated(), indices, act_type
        );
    }

    void backward() override {
        kernel::select_bwd(
            input->get_gradients(), tensor_output.get_linear_output(), tensor_output.get_gradients(), indices, act_type
        );
    }

  private:
    int max_indices;

    Ptr<Operation> input;
    Array<int> indices;
    std::function<int(const Position&)> select_fn;
};

} // namespace nn
