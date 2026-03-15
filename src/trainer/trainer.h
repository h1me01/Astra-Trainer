#pragma once

#include "common.h"
#include "model.h"

namespace trainer {

struct TrainingConfig {
    std::string name = "test";
    int epochs = 100;
    int batch_size = 16384;
    int batches_per_epoch = 6104;
    int save_rate = 20;
    float eval_scale = 400.0f;
};

struct TrainerParams {
    Model& model;
    TrainingConfig config;
    Loss loss;
    Optimizer optim;
    LRScheduler lr_sched;
    WDLScheduler wdl_sched;
    Dataloader dataloader;

    void check() const {
        if (!loss || !optim || !lr_sched || !wdl_sched || !dataloader)
            error("Trainer: All components must be non-null");
    }
};

class Trainer {
  public:
    Trainer(TrainerParams p)
        : model_(p.model),
          config_(p.config),
          loss_(p.loss),
          optim_(p.optim),
          lr_sched_(p.lr_sched),
          wdl_sched_(p.wdl_sched),
          dataloader_(p.dataloader) {

        p.check();
        model_.init(config_.batch_size);
        optim_->init(model_.params());
        dataloader_->init(config_.batch_size);
        targets_ = Array<float>(config_.batch_size, true);
    }

    void set_device(int id) { CUDA_CHECK(cudaSetDevice(id)); }

    void load_checkpoint(const std::string& path);

    void fit(const std::string output_path = "");

  private:
    Model& model_;
    TrainingConfig config_ = {};
    Loss loss_ = nullptr;
    Optimizer optim_ = nullptr;
    LRScheduler lr_sched_ = nullptr;
    WDLScheduler wdl_sched_ = nullptr;
    Dataloader dataloader_ = nullptr;

    int current_epoch_ = 0;
    Array<float> targets_ = {};

    void prepare_batch(const std::vector<TrainingDataEntry>& batch);
    void save_checkpoint(const std::string& path);

    void print_info(int epoch, const std::string output_path) const {
        std::cout << "\n=============================== Training Data ==============================\n\n";
        for (const auto f : dataloader_->filenames())
            std::cout << f << std::endl;

        std::cout << "\n=============================== Trainer Info ===============================\n\n";
        std::cout << "Name          : " << config_.name << std::endl;
        std::cout << "Device        : " << device_Info() << std::endl;
        std::cout << "Epochs        : " << config_.epochs << std::endl;
        std::cout << "Batch Size    : " << config_.batch_size << std::endl;
        std::cout << "Batches/Epoch : " << config_.batches_per_epoch << std::endl;
        std::cout << "Save Rate     : " << config_.save_rate << std::endl;
        std::cout << "LR Scheduler  : " << lr_sched_->info() << std::endl;
        std::cout << "WDL Scheduler : " << wdl_sched_->info() << std::endl;
        std::cout << "Output Path   : " << output_path << std::endl;

        if (epoch > 0)
            std::cout << "\nResuming from epoch " << epoch << " with learning rate " << lr_sched_->get() << std::endl;
    }

    std::string device_Info() const {
        int device = -1;
        CUDA_CHECK(cudaGetDevice(&device));

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        return prop.name;
    }
};

} // namespace trainer
