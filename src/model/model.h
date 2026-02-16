#pragma once

#include <string>
#include <vector>

#include "../dataloader/dataloader.h"
#include "common.h"

using namespace dataloader;

namespace model {

struct TrainingConfig {
    int epochs = 100;
    int batch_size = 16384;
    int batches_per_epoch = 6104;
    int save_rate = 20;
    int thread_count = 2;
    float eval_div = 400.0f;
};

class Model {
  public:
    virtual ~Model() = default;

    void train(const std::string& output_path);

    void load_params(const std::string& file) {
        init();

        FILE* f = fopen(file.c_str(), "rb");
        if (!f)
            error("File " + file + " does not exist!");

        try {
            for (auto& p : network->get_params())
                p->load(f);
            fclose(f);
        } catch (const std::exception& e) {
            fclose(f);
            throw;
        }

        loaded_model = file;
    }

    void save_params(const std::string& file) {
        init();
        save_params_helper(file, false);
    }

    void save_quantized_params(const std::string& file) {
        init();
        save_params_helper(file, true);
    }

    void load_checkpoint(const std::string& checkpoint_path) {
        if (!exists(checkpoint_path))
            error("Checkpoint path does not exist: " + checkpoint_path);

        load_params(checkpoint_path + "/model.bin");
        optim->load(checkpoint_path);

        loaded_checkpoint = checkpoint_path;
    }

    void evaluate_positions(const std::vector<std::string>& positions) {
        init();

        std::cout << "\n================================= Testing ================================\n\n";

        for (const std::string& fen : positions) {
            std::cout << "FEN: " << fen << std::endl;
            std::cout << "Eval: " << predict(fen) << std::endl;
        }
    }

  protected:
    std::string name = "Model";
    TrainingConfig config;

    virtual Operation build(const Input stm_in, const Input nstm_in) = 0;
    virtual int feature_index(PieceType pt, Color pc, Square psq, Square ksq, Color view) = 0;

    virtual bool filter_entry(const TrainingDataEntry& e) { return false; }

    int num_buckets(const std::array<int, 64>& bucket_map) const {
        int max_bucket = 0;
        for (int b : bucket_map)
            max_bucket = std::max(max_bucket, b);
        return max_bucket + 1;
    }

    virtual Loss get_loss() = 0;
    virtual Optimizer get_optim() = 0;
    virtual LRScheduler get_lr_scheduler() = 0;
    virtual WDLScheduler get_wdl_scheduler() = 0;
    virtual std::vector<std::string> get_training_files() = 0;

  private:
    bool is_initialized = false;

    Array<float> targets;

    Loss loss;
    Optimizer optim;
    LRScheduler lr_sched;
    WDLScheduler wdl_sched;
    Input stm_input, nstm_input;

    Ptr<nn::Network> network;
    Ptr<Dataloader> dataloader;

    std::string loaded_model;
    std::string loaded_checkpoint;

    void init();
    void print_info(int epoch, const std::string& output_path) const;
    void fill_inputs(std::vector<TrainingDataEntry>& ds);

    float predict(const std::string& fen);

    void save_checkpoint(const std::string& path) {
        try {
            create_directories(path);
        } catch (const filesystem_error& e) {
            error("Failed creating directory " + path + ": " + e.what());
        }

        save_params(path + "/model.bin");
        save_quantized_params(path + "/quantized_model.nnue");

        if (optim)
            optim->save(path);

        std::cout << "Saved checkpoint to " << path << std::endl;
    }

    int epoch_from_checkpoint(const std::string& checkpoint_name) const {
        size_t dash_pos = checkpoint_name.find_last_of('_');
        if (dash_pos == std::string::npos) {
            std::cout << "Could not parse epoch from checkpoint name, starting from epoch 0\n";
            return 0;
        }

        std::string epoch_str = checkpoint_name.substr(dash_pos + 1);
        if (epoch_str == "final") {
            std::cout << "Loading from final checkpoint, starting new training cycle\n";
            return 0;
        }

        try {
            return std::stoi(epoch_str);
        } catch (...) {
            std::cout << "Could not parse epoch from checkpoint name, starting from epoch 0\n";
            return 0;
        }
    }

    void save_params_helper(const std::string& file, bool quantized) {
        FILE* f = fopen(file.c_str(), "wb");
        if (!f)
            error("Failed writing weights to " + file);

        try {
            for (auto& p : network->get_params()) {
                if (quantized)
                    p->save_quantized(f);
                else
                    p->save(f);
            }
            fclose(f);
        } catch (const std::exception& e) {
            fclose(f);
            throw;
        }
    }
};

} // namespace model
