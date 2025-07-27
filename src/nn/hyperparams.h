#pragma once

#include <vector>

#include "data/include.h"
#include "layers/include.h"
#include "loss/include.h"
#include "optimizer/include.h"

struct Hyperparams {
    std::stringstream info;

    int epochs = 500;
    int batch_size = 16384;
    int batches_per_epoch = 6104;
    int save_rate = 10;
    int thread_count = 1; // dataloader thread count
    float output_scalar = 400.0f;
    float start_lambda = 0.7f;
    float end_lambda = 0.8f;

    std::string loaded_weights = "";
    std::string loaded_checkpoint = "";

    Loss *loss = nullptr;
    Optimizer *optim = nullptr;

    std::vector<LayerBase *> layers = {};
    std::array<int, 64> input_bucket = {};

    Hyperparams() {}

    Hyperparams(               //
        int epochs,            //
        int batch_size,        //
        int batches_per_epoch, //
        int save_rate,         //
        int thread_count,      //
        float output_scalar,   //
        float start_lambda,    //
        float end_lambda       //
    ) {
        this->epochs = epochs;
        this->batch_size = batch_size;
        this->batches_per_epoch = batches_per_epoch;
        this->save_rate = save_rate;
        this->thread_count = thread_count;
        this->output_scalar = output_scalar;
        this->start_lambda = start_lambda;
        this->end_lambda = end_lambda;
    }

    Hyperparams(const Hyperparams &other) {
        this->epochs = other.epochs;
        this->batch_size = other.batch_size;
        this->batches_per_epoch = other.batches_per_epoch;
        this->save_rate = other.save_rate;
        this->thread_count = other.thread_count;
        this->output_scalar = other.output_scalar;
        this->start_lambda = other.start_lambda;
        this->end_lambda = other.end_lambda;

        this->loaded_weights = other.loaded_weights;
        this->loaded_checkpoint = other.loaded_checkpoint;

        this->loss = other.loss;
        this->optim = other.optim;

        this->layers = other.layers;
        this->input_bucket = other.input_bucket;
    }

    void print_info() {
        info << "\n================================= Network Info =================================\n\n";
        info << "Epochs:            " << epochs << std::endl;
        info << "Batch Size:        " << batch_size << std::endl;
        info << "Batches/Epoch:     " << batches_per_epoch << std::endl;
        info << "Save Rate:         " << save_rate << std::endl;
        info << "Output Scalar:     " << output_scalar << std::endl;
        info << "Start Lambda:      " << start_lambda << std::endl;
        info << "End Lambda:        " << end_lambda << std::endl;
        info << "Loss:              " << loss->get_info() << std::endl;
        info << "Optimizer:         " << optim->get_info() << std::endl;
        info << "LR Scheduler:      " << optim->get_lr_scheduler_info() << std::endl;
        if(!loaded_checkpoint.empty())
            info << "Loaded Checkpoint: " << loaded_checkpoint << std::endl;
        else if(!loaded_weights.empty())
            info << "Loaded Weights:    " << loaded_weights << std::endl;

        info << "\n============================= Network Architecture =============================\n\n";
        info << "Input Bucket: " << std::endl;
        for(size_t i = 0; i < input_bucket.size(); ++i) {
            info << std::setw(3) << input_bucket[i];
            if((i + 1) % 8 == 0)
                info << "\n";
        }

        info << "\nLayers:" << std::endl;
        for(const auto &l : layers)
            info << " -> " << l->get_info();

        std::cout << info.str();
        std::cout << "\n================================================================================\n";
    }
};
