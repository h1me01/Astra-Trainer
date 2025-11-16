#pragma once

#include <string>
#include <vector>

#include "../nn/types.h"
#include "../training_data_formats/include.h"

namespace nn {
class Trainer;
class Input;
class LayerBase;
class Loss;
class Optimizer;
class Trainer;
} // namespace nn

using namespace nn;

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

    virtual LayerPtr build(const InputPtr &stm_in, const InputPtr &nstm_in) = 0;

    void load_weights(const std::string &file);
    void save_weights(const std::string &file);

    void evaluate_positions(const std::vector<std::string> &positions);
    void train(std::vector<std::string> data_path, std::string output_path, std::string checkpoint_name);

    virtual int feature_index(PieceType pt, Color pc, Square psq, Square ksq, Color view) = 0;

    virtual LossPtr get_loss() {
        return nullptr;
    }

    virtual OptimizerPtr get_optim() {
        return nullptr;
    }

    virtual LRSchedulerPtr get_lr_scheduler() {
        return nullptr;
    }

    std::string get_name() const {
        return name;
    }

    HyperParams get_params() const {
        return params;
    }

  protected:
    HyperParams params;

    std::string name = "";

    template <typename T, typename... Args> //
    auto make(Args &&...args) {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }

  private:
    std::unique_ptr<Trainer> trainer;

    void init_trainer();
};

} // namespace model
