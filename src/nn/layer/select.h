#pragma once

#include "layer.h"

namespace nn {

class Select : public Layer {
  public:
    explicit Select(const Ptr<Layer>& input, const std::function<int(const Position&)>& select_fn)
        : input(input),
          select_fn(select_fn) {}

    void init(int batch_size) override {
        Position start_pos = Position::startPosition();
        max_indices = select_fn(start_pos) + 1;
        indices = Array<int>(batch_size);

        input_size = input->get_output_size();
        if (input_size % max_indices != 0)
            error("Select layer input size must be divisible by select size!");
        output_size = input->get_output_size() / max_indices;

        Layer::init(batch_size);
    }

    void step(const std::vector<TrainingDataEntry>& data_entries) override {
        Layer::step(data_entries);

        for (int i = 0; i < (int)data_entries.size(); i++) {
            int idx = select_fn(data_entries[i].pos);
            if (idx < 0 || idx >= max_indices)
                error("Index function of Select returned invalid index!");
            indices(i) = idx;
        }

        indices.host_to_dev();
    }

    void forward() override {
        kernel::select_fwd(input->get_output(), output.get_linear_output(), output.get_activated(), indices, act_type);
    }

    void backward() override {
        kernel::select_bwd(
            input->get_gradients(), output.get_linear_output(), output.get_gradients(), indices, act_type
        );
    }

    std::vector<Ptr<Layer>> get_inputs() override {
        return {input};
    }

  private:
    int max_indices;

    Ptr<Layer> input;
    Array<int> indices;
    std::function<int(const Position&)> select_fn;
};

} // namespace nn
