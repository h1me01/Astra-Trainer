#pragma once

#include <fstream>
#include <vector>

#include "../dataloader/dataloader.h"
#include "data/include.h"
#include "layers/include.h"
#include "loss.h"
#include "optimizer/include.h"

struct Hyperparams {
    int epochs = 500;
    int batch_size = 16384;
    int batches_per_epoch = 6104;
    int save_rate = 10;
    int thread_count = 1; // dataloader thread count
    float output_scalar = 400.0f;
    float start_lambda = 0.7f;
    float end_lambda = 0.8f;

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
};

class Network {
  private:
    bool is_initialized = false;

    Hyperparams hp;

    std::stringstream info;

    std::vector<LayerBase *> layers;
    std::array<int, 64> input_bucket = {};

    Loss *loss = nullptr;
    Optimizer *optim = nullptr;

    Array<float> targets;

    void init() {
        if(is_initialized)
            return;

        if(layers.empty())
            error("No hidden layers set for the network.");
        if(optim == nullptr)
            error("Optimizer is not set for the network.");
        if(loss == nullptr)
            error("Loss function is not set for the network.");

        optim->init(layers);
        for(LayerBase *l : layers)
            l->init(hp.batch_size);

        is_initialized = true;
    }

    void print_info() {
        std::string loaded_weights_path = info.str();

        info.str(""); // clear the stringstream
        info << "\n================================= Network Info =================================\n\n";
        info << "Epochs:         " << hp.epochs << std::endl;
        info << "Batch Size:     " << hp.batch_size << std::endl;
        info << "Batches/Epoch:  " << hp.batches_per_epoch << std::endl;
        info << "Save Rate:      " << hp.save_rate << std::endl;
        info << "Output Scalar:  " << hp.output_scalar << std::endl;
        info << "Start Lambda:   " << hp.start_lambda << std::endl;
        info << "End Lambda:     " << hp.end_lambda << std::endl;
        info << "Loss:           " << loss->info() << std::endl;
        info << "Optimizer:      " << optim->get_info() << std::endl;
        info << "LR Scheduler:   " << optim->get_lr_scheduler_info() << std::endl;
        if(!loaded_weights_path.empty())
            info << "Loaded Weights: " << loaded_weights_path << std::endl;

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

        std::cout << "\n================================================================================\n\n";
    }

    void forward() {
        for(size_t i = 0; i < layers.size(); i++)
            layers[i]->forward();
    }

    void backward() {
        for(int i = layers.size() - 1; i >= 0; i--)
            layers[i]->backward();
    }

    void save_checkpoint(const std::string &path);

    int index(PieceType pt, Color pc, Square psq, Square ksq, Color view);

    void fill(std::vector<DataEntry> &ds, float lambda);

  public:
    explicit Network(Hyperparams hp) : hp(hp), targets(hp.batch_size) {
        create_cublas();
    }

    ~Network() {
        destroy_cublas();
    }

    void load_weights(const std::string &file) {
        std::ifstream f(file, std::ios::binary);

        // check if the file exists
        if(!f)
            error("File " + file + " does not exist");

        try {
            for(LayerBase *l : layers) {
                for(Tensor *t : l->get_params()) {
                    DenseMatrix<float> &weights = t->get_data();

                    f.read(reinterpret_cast<char *>(weights.host_address()), weights.size() * sizeof(float));
                    if(f.gcount() != static_cast<std::streamsize>(weights.size() * sizeof(float))) {
                        error("insufficient data read from file. Expected " + //
                              std::to_string(weights.size()) + " floats");
                    }

                    weights.host_to_dev();
                }
            }

            info << file;
        } catch(const std::exception &e) {
            error("Failed loading weights from " + file + ": " + e.what());
        }
    }

    // assumes output activation is sigmoid
    float predict(std::string fen) {
        init();

        Position pos;
        pos.set(fen);

        DataEntry e;
        e.pos = pos;

        std::vector<DataEntry> ds{e};

        fill(ds, 1);
        forward();

        LayerBase *output_layer = layers.back();

        DenseMatrix<float> &output = get_output().get_data();
        output.dev_to_host();

        return output(0) * hp.output_scalar;
    }

    void evaluate_positions(const std::vector<std::string> &positions) {
        std::cout << "\n================================ Testing Network ===============================\n\n";

        for(const std::string &fen : positions) {
            std::cout << "FEN: " << fen << std::endl;
            std::cout << "Eval: " << predict(fen) << std::endl;
        }
    }

    void set_loss(Loss *loss) {
        this->loss = loss;
    }

    void set_optim(Optimizer *optim) {
        this->optim = optim;
    }

    void set_input_bucket(std::array<int, 64> input_bucket) {
        this->input_bucket = input_bucket;
    }

    void set_layers(std::vector<LayerBase *> hidden_layers) {
        this->layers = hidden_layers;
    }

    int get_batch_size() {
        return hp.batch_size;
    }

    Tensor &get_output() {
        return layers.back()->get_output().activated;
    }

    std::vector<LayerBase *> get_layers() {
        return layers;
    }

    void train(std::vector<std::string> data_path, std::string output_path, std::string checkpoint_name);
};
