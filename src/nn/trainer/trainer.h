#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>

#include "../../models/model.h"
#include "../batch_data/batch_data.h"
#include "../loss/include.h"
#include "../lr_scheduler/include.h"
#include "../network/network.h"
#include "../optimizer/include.h"

using namespace model;

namespace nn {

class Trainer {
  public:
    explicit Trainer(Model *model) : model(model) {
        params = model->get_params();
        network = std::make_unique<Network>();

        this->loss = model->get_loss();
        this->optim = model->get_optim();
        this->lr_sched = model->get_lr_scheduler();

        if(this->loss == nullptr)
            error("Loss function is not set for the trainer!");
        if(this->optim == nullptr)
            error("Optimizer is not set for the trainer!");
        if(this->lr_sched == nullptr)
            error("LR Scheduler is not set for the trainer!");

        network->set_feature_index_fn( //
            [this](PieceType pt, Color pc, Square psq, Square ksq, Color view) {
                return this->model->feature_index(pt, pc, psq, ksq, view);
            });

        network->set_output_layer(model->build( //
            network->get_inputs().first,
            network->get_inputs().second));

        if(optim == nullptr)
            error("Optimizer is not set for the trainer!");
        if(loss == nullptr)
            error("Loss function is not set for the trainer!");

        network->init(params.batch_size);
        optim->init(network->get_layers());
    }

    virtual ~Trainer() = default;

    void train(std::vector<std::string> data_path, std::string output_path, std::string checkpoint_name);

    void load_weights(const std::string &file) {
        network->load_weights(file);
        loaded_weights = file;
    }

    void save_weights(const std::string &file) {
        network->save_weights(file);
    }

    void evaluate_positions(const std::vector<std::string> &positions) {
        std::cout << "\n================================= Testing ================================\n\n";

        for(const std::string &fen : positions) {
            std::cout << "FEN: " << fen << std::endl;
            std::cout << "Eval: " << predict(fen) << std::endl;
        }
    }

  private:
    Model *model;
    HyperParams params;

    LossPtr loss;
    OptimizerPtr optim;
    LRSchedulerPtr lr_sched;
    std::unique_ptr<Network> network;

    std::string loaded_weights = "";
    std::string loaded_checkpoint = "";

    void save_checkpoint(const std::string &path);

    void print_info(std::string output_path) const {
        std::cout << "\n============================== Trainer Info ==============================\n\n";
        std::cout << "Model name:    " << model->get_name() << std::endl;
        std::cout << "Epochs:        " << params.epochs << std::endl;
        std::cout << "Batch Size:    " << params.batch_size << std::endl;
        std::cout << "Batches/Epoch: " << params.batches_per_epoch << std::endl;
        std::cout << "Save Rate:     " << params.save_rate << std::endl;
        std::cout << "Thread Count:  " << params.thread_count << std::endl;
        std::cout << "Learning Rate: " << params.lr << std::endl;
        std::cout << "Eval Div:      " << params.eval_div << std::endl;
        std::cout << "Lambda Start:  " << params.lambda_start << std::endl;
        std::cout << "Lambda End:    " << params.lambda_end << std::endl;
        std::cout << "Output Path:   " << output_path << std::endl;
        if(!loaded_checkpoint.empty())
            std::cout << "Loaded Checkpoint: " << loaded_checkpoint << std::endl;
        else if(!loaded_weights.empty())
            std::cout << "Loaded Weights:    " << loaded_weights << std::endl;
    }

    float predict(std::string fen) {
        Position pos;
        pos.set(fen);

        DataEntry e;
        e.pos = pos;

        std::vector<DataEntry> ds{e};

        batch_data::data_entries = ds;
        network->fill_inputs(ds, 1.0f, params.eval_div);
        network->forward();

        auto &output = network->get_output().get_values();
        output.dev_to_host();

        return output(0) * params.eval_div;
    }
};

} // namespace nn
