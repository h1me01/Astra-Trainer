#pragma once

#include <unordered_set>

#include "../layer/include.h"

namespace nn {

class Network {
  public:
    Network() {
        kernel::create_cublas();

        inputs.first = std::make_unique<Input>(32);
        inputs.second = std::make_unique<Input>(32);
    }

    ~Network() {
        kernel::destroy_cublas();
    }

    void load_weights(const std::string &file);
    void save_weights(const std::string &path);

    void save_quantized_weights(const std::string &path);

    void fill_inputs(std::vector<DataEntry> &ds, float lambda, float eval_div);

    void init(int batch_size) {
        if(architecture.empty())
            error("No layers set for the network!");

        inputs.first->init(batch_size);
        inputs.second->init(batch_size);

        // set output layer will initialize layers vector
        // in reverse order, so we need to reverse it to get correct order
        std::reverse(architecture.begin(), architecture.end());

        for(auto &l : architecture)
            l->init(batch_size);

        targets = Array<float>(batch_size);
        batch_data::data_entries.reserve(batch_size);
    }

    void forward() {
        for(auto &l : architecture)
            l->step();

        for(size_t i = 0; i < architecture.size(); i++)
            architecture[i]->forward();
    }

    void backward() {
        for(int i = architecture.size() - 1; i >= 0; i--)
            architecture[i]->backward();
    }

    void set_output_layer(const Ptr<Layer> &output_layer) {
        if(output_layer == nullptr)
            error("Output layer is null!");

        architecture.clear();
        architecture.push_back(output_layer);
        init_layers(output_layer->get_inputs());
    }

    void set_feature_index_fn(std::function<int(PieceType, Color, Square, Square, Color)> fn) {
        this->feature_index_fn = fn;
    }

    std::pair<Ptr<Input>, Ptr<Input>> &get_inputs() {
        return inputs;
    }

    Array<float> &get_targets() {
        return targets;
    }

    Tensor<float> &get_output() {
        return architecture.back()->get_output();
    }

    std::vector<Ptr<Layer>> get_layers() {
        std::vector<Ptr<Layer>> main_layers;
        std::unordered_set<Layer *> seen;

        for(auto &l : architecture) {
            auto m = l->get_main();
            if(m && seen.insert(m.get()).second)
                main_layers.push_back(m);
        }

        return main_layers;
    }

  private:
    Array<float> targets;

    std::vector<Ptr<Layer>> architecture;
    std::pair<Ptr<Input>, Ptr<Input>> inputs;

    std::function<int(PieceType, Color, Square, Square, Color)> feature_index_fn;

    void init_layers(const std::vector<Ptr<Layer>> &layers) {
        if(layers.empty())
            return;

        for(const auto &l : layers) {
            architecture.push_back(l);
            init_layers(l->get_inputs());
        }
    }
};

} // namespace nn
