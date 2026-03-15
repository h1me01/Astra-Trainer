#pragma once

#include <typeinfo>

#include "builder.h"
#include "common.h"

namespace trainer {

class Model {
  public:
    struct Inputs {
        std::vector<nn::op::Input*> ptrs;
        Input& operator[](size_t i) const { return *ptrs[i]; }
        size_t size() const { return ptrs.size(); }
    };

    using GraphBuilder = std::function<Node()>;
    using InputFiller = std::function<void(const std::vector<TrainingDataEntry>&, Inputs&)>;

    virtual ~Model() = default;

    void set_graph(GraphBuilder fn) { graph_builder_ = std::move(fn); }
    void set_input_filler(InputFiller fn) { input_filler_ = std::move(fn); }

    void init(int batch_size) { network().init(batch_size); }

    Node build_graph() {
        if (!graph_builder_)
            error("Model: no graph builder set (call set_graph or override build_graph)");
        return graph_builder_();
    }

    void prepare_inputs(const std::vector<TrainingDataEntry>& batch) {
        for (auto& input : network().get_inputs())
            input->reset();
        fill_inputs(batch);
    }

    void forward(const std::vector<TrainingDataEntry>& batch) { network().forward(batch); }
    void backward() { network().backward(); }

    void save_params(const std::string& file) { save_params_helper(file, false); }
    void save_quantized_params(const std::string& file) { save_params_helper(file, true); }

    void load_params(const std::string& file) {
        FILE* f = fopen(file.c_str(), "rb");
        if (!f)
            error("Model: File " + file + " does not exist!");

        try {
            for (auto& p : net_->get_params())
                p->load(f);
            fclose(f);
            std::cout << "Model: Loaded parameters from " << file << std::endl;
        } catch (const std::exception& e) {
            fclose(f);
            error("Model: Failed loading parameters from " + file + ": " + e.what());
        }
    }

    float predict(const std::string& fen) {
        network().init(1);

        Position pos;
        pos.set(fen);
        std::vector<TrainingDataEntry> ds{{pos}};

        prepare_inputs(ds);
        inputs_to_dev();

        forward(ds);

        auto& output = network().get_output().get_data();
        output.dev_to_host();

        return output(0);
    }

    void inputs_to_dev() {
        for (auto& input : network().get_inputs())
            input->get_indices().host_to_dev();
    }

    Tensor& get_output() { return network().get_output(); }
    std::vector<nn::param::Param*> get_params() { return network().get_params(); }

  private:
    GraphBuilder graph_builder_;
    InputFiller input_filler_;

    Inputs inputs_;
    std::unique_ptr<nn::Network> net_;

    nn::Network& network() {
        if (!net_) {
            net_ = std::make_unique<nn::Network>(build_graph());
            inputs_.ptrs.clear();
            for (auto* input : network().get_inputs())
                inputs_.ptrs.push_back(input);
        }
        return *net_;
    }

    void fill_inputs(const std::vector<TrainingDataEntry>& batch) {
        if (!input_filler_)
            error("Model: no input filler set (call set_input_filler or override fill_inputs)");
        input_filler_(batch, inputs_);
    }

    void save_params_helper(const std::string& file, bool quantized) {
        FILE* f = fopen(file.c_str(), "wb");
        if (!f)
            error("Model: Failed writing weights to " + file);

        try {
            for (auto& p : net_->get_params()) {
                if (quantized)
                    p->save_quantized(f);
                else
                    p->save(f);
            }
            fclose(f);
        } catch (const std::exception& e) {
            fclose(f);
            error("Model: Failed writing weights to " + file + ": " + e.what());
        }
    }
};

} // namespace trainer
