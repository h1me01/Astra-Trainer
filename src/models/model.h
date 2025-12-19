#pragma once

#include <string>
#include <vector>

#include "../dataloader/dataloader.h"
#include "../nn/include.h"

using namespace dataloader;

namespace model {

struct HyperParams {
    int epochs;
    int batch_size;
    int batches_per_epoch;
    int save_rate;
    int thread_count;
    float lr;
    float eval_div;
    float lambda_start;
    float lambda_end;
};

class Model {
  public:
    Model(std::string name) : name(name) {}
    virtual ~Model() = default;

    void train(std::string output_path, std::string checkpoint_name = "");

    void load_weights(const std::string &file) {
        init();
        network->load_weights(file);
        loaded_weights = file;
    }

    void save_weights(const std::string &file) {
        init();
        network->save_weights(file);
    }

    void save_quantized_weights(const std::string &file) {
        init();
        network->save_quantized_weights(file);
    }

    void evaluate_positions(const std::vector<std::string> &positions) {
        init();

        std::cout << "\n================================= Testing ================================\n\n";

        for(const std::string &fen : positions) {
            std::cout << "FEN: " << fen << std::endl;
            std::cout << "Eval: " << predict(fen) << std::endl;
        }
    }

  protected:
    HyperParams params;

    template <typename T, typename... Args> //
    auto make(Args &&...args) {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }

    virtual Ptr<Layer> build(const Ptr<Input> &stm_in, const Ptr<Input> &nstm_in) = 0;

    virtual int feature_index(PieceType pt, Color pc, Square psq, Square ksq, Color view) = 0;

    int num_buckets(std::array<int, 64> bucket_map) {
        int max_bucket = 0;
        for(int b : bucket_map)
            if(b > max_bucket)
                max_bucket = b;
        return max_bucket + 1;
    }

    virtual Ptr<Loss> get_loss() {
        return nullptr;
    }

    virtual Ptr<Optimizer> get_optim() {
        return nullptr;
    }

    virtual Ptr<LRScheduler> get_lr_scheduler() {
        return nullptr;
    }

    virtual Ptr<Dataloader> get_dataloader() {
        return nullptr;
    }

  private:
    std::string name = "";
    bool is_initialized = false;

    Array<float> targets;

    Ptr<Loss> loss;
    Ptr<Optimizer> optim;
    Ptr<LRScheduler> lr_sched;
    Ptr<Dataloader> dataloader;
    Ptr<Input> stm_input, nstm_input;
    std::unique_ptr<Network> network;

    std::string loaded_weights = "";
    std::string loaded_checkpoint = "";

    void fill_inputs(std::vector<TrainingDataEntry> &ds, float lambda);

    void init() {
        if(is_initialized)
            return;

        targets = Array<float>(params.batch_size);

        network = std::make_unique<Network>();
        stm_input = std::make_shared<Input>(32);
        nstm_input = std::make_shared<Input>(32);

        loss = get_loss();
        optim = get_optim();
        lr_sched = get_lr_scheduler();
        dataloader = get_dataloader();

        if(loss == nullptr)
            error("Loss function is not set for the trainer!");
        if(optim == nullptr)
            error("Optimizer is not set for the trainer!");
        if(lr_sched == nullptr)
            error("LR Scheduler is not set for the trainer!");
        if(dataloader == nullptr)
            error("Dataloader is not set for the trainer!");

        network->set_output_layer(build(stm_input, nstm_input));

        network->init(params.batch_size);
        stm_input->init(params.batch_size);
        nstm_input->init(params.batch_size);
        optim->init(network->get_layers());

        is_initialized = true;
    }

    void print_info(int epoch, std::string output_path) const {
        std::cout << "\n=============================== Training Data ==============================\n\n";
        const auto &training_files = dataloader->get_filenames();
        if(training_files.empty())
            error("No training data found in the specified paths!");

        for(const auto &f : training_files)
            std::cout << f << std::endl;

        std::cout << "\n=============================== Trainer Info ===============================\n\n";
        std::cout << "Model name:        " << name << std::endl;
        std::cout << "Epochs:            " << params.epochs << std::endl;
        std::cout << "Batch Size:        " << params.batch_size << std::endl;
        std::cout << "Batches/Epoch:     " << params.batches_per_epoch << std::endl;
        std::cout << "Save Rate:         " << params.save_rate << std::endl;
        std::cout << "Thread Count:      " << params.thread_count << std::endl;
        std::cout << "Learning Rate:     " << params.lr << std::endl;
        std::cout << "Eval Div:          " << params.eval_div << std::endl;
        std::cout << "Lambda Start:      " << params.lambda_start << std::endl;
        std::cout << "Lambda End:        " << params.lambda_end << std::endl;
        std::cout << "Output Path:       " << output_path << std::endl;

        if(!loaded_checkpoint.empty())
            std::cout << "Loaded Checkpoint: " << loaded_checkpoint << std::endl;
        else if(!loaded_weights.empty())
            std::cout << "Loaded Weights:    " << loaded_weights << std::endl;

        if(epoch > 0)
            std::cout << "\nResuming from epoch " << epoch << " with learning rate " << lr_sched->get_lr() << std::endl;
    }

    float predict(std::string fen) {
        Position pos;
        pos.set(fen);

        TrainingDataEntry e;
        e.pos = pos;

        std::vector<TrainingDataEntry> ds{e};

        fill_inputs(ds, 1.0f);
        network->forward(ds);

        auto &output = network->get_output().get_output();
        output.dev_to_host();

        return output(0) * params.eval_div;
    }

    void save_checkpoint(const std::string &path) {
        try {
            create_directories(path);
        } catch(const filesystem_error &e) {
            error("Failed creating directory " + path + ": " + e.what());
        }

        network->save_weights(path + "/weights.bin");
        network->save_quantized_weights(path + "/qweights.nnue");

        if(optim != nullptr)
            optim->save(path);

        std::cout << "Saved checkpoint to " << path << std::endl;
    }

    int epoch_from_checkpoint(const std::string &checkpoint_name) {
        size_t dash_pos = checkpoint_name.find_last_of('_');
        if(dash_pos == std::string::npos) {
            std::cout << "Could not parse epoch from checkpoint name, starting from epoch 0\n";
            return 0;
        }

        std::string epoch_str = checkpoint_name.substr(dash_pos + 1);
        if(epoch_str == "final") {
            std::cout << "Loading from final checkpoint, starting new training cycle\n";
            return 0;
        }

        try {
            int parsed_epoch = std::stoi(epoch_str);
            return parsed_epoch;
        } catch(...) {
            std::cout << "Could not parse epoch from checkpoint name, starting from epoch 0\n";
            return 0;
        }
    }
};

} // namespace model
