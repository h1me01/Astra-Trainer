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
    virtual ~Model() = default;

    void train(const std::string& output_path, const std::string& checkpoint_name = "");

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

    void evaluate_positions(const std::vector<std::string>& positions) {
        init();

        std::cout << "\n================================= Testing ================================\n\n";

        for (const std::string& fen : positions) {
            std::cout << "FEN: " << fen << std::endl;
            std::cout << "Eval: " << predict(fen) << std::endl;
        }
    }

  protected:
    std::string name;
    HyperParams params;

    template <typename T, typename... Args>
    auto make(Args&&... args) {
        auto op = std::make_shared<T>(std::forward<Args>(args)...);

        if constexpr (std::is_base_of_v<Operation, T>)
            network->add_operation(op);

        return op;
    }

    virtual void build(const Ptr<Input>& stm_in, const Ptr<Input>& nstm_in) = 0;
    virtual int feature_index(PieceType pt, Color pc, Square psq, Square ksq, Color view) = 0;

    virtual bool filter_entry(const TrainingDataEntry& e) {
        return false;
    }

    int num_buckets(const std::array<int, 64>& bucket_map) const {
        int max_bucket = 0;
        for (int b : bucket_map)
            max_bucket = std::max(max_bucket, b);
        return max_bucket + 1;
    }

    virtual Ptr<Loss> get_loss() = 0;
    virtual Ptr<Optimizer> get_optim() = 0;
    virtual Ptr<LRScheduler> get_lr_scheduler() = 0;
    virtual std::vector<std::string> get_training_files() = 0;

  private:
    bool is_initialized = false;

    Array<float> targets;

    Ptr<Loss> loss;
    Ptr<Optimizer> optim;
    Ptr<LRScheduler> lr_sched;
    Ptr<Input> stm_input, nstm_input;

    std::unique_ptr<Network> network;
    std::unique_ptr<Dataloader> dataloader;

    std::string loaded_model;
    std::string loaded_checkpoint;

    void init();
    void print_info(int epoch, const std::string& output_path) const;
    void fill_inputs(std::vector<TrainingDataEntry>& ds, float lambda);

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
