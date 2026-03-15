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

class Trainer {
  public:
    Model& model;
    TrainingConfig config = {};
    Loss loss = nullptr;
    Optimizer optimiser = nullptr;
    LRScheduler lr_sched = nullptr;
    WDLScheduler wdl_sched = nullptr;
    Dataloader dataloader = nullptr;

    int current_epoch_ = 0;
    Array<float> targets_ = {};
    bool initialized_ = false;

    void set_device(int id) { CUDA_CHECK(cudaSetDevice(id)); }

    void load_checkpoint(const std::string& path);

    void fit(const std::string output_path = "");

  private:
    void init();
    void prepare_batch(const std::vector<TrainingDataEntry>& batch);
    void save_checkpoint(const std::string& path);

    void print_info(int epoch, const std::string output_path) const {
        std::cout << "\n=============================== Training Data ==============================\n\n";
        for (const auto f : dataloader->filenames())
            std::cout << f << std::endl;

        std::cout << "\n=============================== Trainer Info ===============================\n\n";
        std::cout << "Name          : " << config.name << std::endl;
        std::cout << "Device        : " << device_Info() << std::endl;
        std::cout << "Epochs        : " << config.epochs << std::endl;
        std::cout << "Batch Size    : " << config.batch_size << std::endl;
        std::cout << "Batches/Epoch : " << config.batches_per_epoch << std::endl;
        std::cout << "Save Rate     : " << config.save_rate << std::endl;
        std::cout << "LR Scheduler  : " << lr_sched->info() << std::endl;
        std::cout << "WDL Scheduler : " << wdl_sched->info() << std::endl;
        std::cout << "Output Path   : " << output_path << std::endl;

        if (epoch > 0)
            std::cout << "\nResuming from epoch " << epoch << " with learning rate " << lr_sched->get() << std::endl;
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
