#pragma once

#include "ops.h"

namespace nn {

class Select : public Operation {
  public:
    Select(Ptr<Operation> input, const std::function<int(const Position&)>& fn)
        : input(input),
          fn(fn) {

        max_indices = fn(Position::startPosition()) + 1;

        input_dim = input->get_output_dim();
        output_dim = input_dim / max_indices;

        if (input_dim % max_indices != 0)
            error("Select input dimension must be a multiple of select size!");
    }

    void init(int batch_size) override {
        Operation::init(batch_size);
        indices = Array<int>(batch_size, true);
    }

    void step(const std::vector<TrainingDataEntry>& data_entries) override {
        Operation::step(data_entries);

        for (int i = 0; i < (int)data_entries.size(); i++) {
            int idx = fn(data_entries[i].pos);
            if (idx < 0 || idx >= max_indices)
                error("Index function of Select returned invalid index!");
            indices(i) = idx;
        }

        indices.host_to_dev();
    }

    void forward() override { kernel::select_fwd(input->get_data(), output.get_data(), indices, act_type); }

    void backward() override { kernel::select_bwd(input->get_grads(), output, indices, act_type); }

    std::vector<Ptr<Operation>> get_inputs() const override { return {input}; }

  private:
    int max_indices;

    Ptr<Operation> input;
    Array<int> indices;
    std::function<int(const Position&)> fn;
};

} // namespace nn
