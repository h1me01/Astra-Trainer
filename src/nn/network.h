#pragma once

#include <fstream>
#include <functional>
#include <vector>

#include "../dataloader/dataloader.h"
#include "data.h"
#include "layers/bucketed.h"
#include "layers/feature_transformer.h"
#include "layers/fully_connected.h"
#include "loss.h"
#include "optimizer.h"

class Network {
  private:
    bool is_initialized = false;

    int Epochs;
    int BatchSize;
    int BatchesPerEpoch;
    int SaveRate;
    int ThreadCount; // dataloader thread count

    float OutputScalar;
    float StartLambda;
    float EndLambda;

    std::stringstream info;

    std::vector<LayerBase *> layers;
    std::array<int, 64> input_bucket;

    Loss *loss = nullptr;
    Optimizer *optim = nullptr;

    Array<float> targets;

    std::function<void(FILE *)> quantFunc;

    void init() {
        if(is_initialized)
            return;

        ASSERT(!layers.empty());
        ASSERT(quantFunc != nullptr);
        ASSERT(optim != nullptr && loss != nullptr);

        optim->init(layers);
        for(LayerBase *l : layers)
            l->init(BatchSize);

        is_initialized = true;
    }

    void print_info() {
        // save and print network info
        info << "\n================================= Network Info =================================\n\n";
        info << "Epochs: " << Epochs << std::endl;
        info << "Batch Size: " << BatchSize << std::endl;
        info << "Batches per Epoch: " << BatchesPerEpoch << std::endl;
        info << "Save Rate: " << SaveRate << std::endl;
        info << "Output Scalar: " << OutputScalar << std::endl;
        info << "Start Lambda: " << StartLambda << std::endl;
        info << "End Lambda: " << EndLambda << std::endl;
        info << "Loss: " << loss->info() << std::endl;
        info << "Optimizer: " << optim->get_info() << std::endl;

        info << "\n============================= Network Architecture =============================\n\n";
        info << "King Bucket: " << std::endl;
        for(size_t i = 0; i < input_bucket.size(); ++i) {
            info << std::setw(3) << input_bucket[i];
            if((i + 1) % 8 == 0)
                info << "\n";
        }

        info << "\nHidden Layers:" << std::endl;
        for(LayerBase *l : layers)
            info << " - " << l->get_info();
        info << "\n";

        std::cout << info.str();
    }

    void forward() {
        for(size_t i = 0; i < layers.size(); i++)
            layers[i]->forward();
    }

    void backward() {
        for(int i = layers.size() - 1; i >= 0; i--)
            layers[i]->backward();
    }

    void save_checkpoint(const std::string &path) {
        ASSERT(quantFunc != nullptr);

        // create directory if it doesn't exist
        try {
            std::filesystem::create_directories(path);
        } catch(const std::filesystem::filesystem_error &e) {
            throw std::runtime_error("Failed to create directory " + path + ": " + e.what());
        }

        // save weights
        try {
            const std::string file = path + "/weights.bin";
            FILE *f = fopen(file.c_str(), "wb");
            if(!f)
                throw std::runtime_error("Failed to write weights to " + file);

            for(LayerBase *l : layers) {
                for(Tensor *t : l->get_params()) {
                    DenseMatrix &weights = t->get_vals();
                    weights.dev_to_host();

                    int written = fwrite(weights.host_address(), sizeof(float), weights.size(), f);
                    if(written != weights.size())
                        throw std::runtime_error("Error writing weights to file");
                }
            }

            fclose(f);
        } catch(const std::exception &e) {
            throw std::runtime_error(std::string("Failed to save weights: ") + e.what());
        }

        // save quantized weights
        try {
            FILE *f = fopen((path + "/qweights.net").c_str(), "wb");
            if(!f)
                throw std::runtime_error("Failed to write quantized weights");

            quantFunc(f);

            fclose(f);
        } catch(const std::exception &e) {
            throw std::runtime_error(std::string("Failed to save quantized weights: ") + e.what());
        }

        // save optimizer state
        if(optim != nullptr)
            optim->save(path);

        std::cout << "Saved checkpoint" << std::endl;
    }

    int index(PieceType pt, Color pc, Square psq, Square ksq, Color view);

    void fill(std::vector<DataEntry> &ds, float lambda);

  public:
    // clang-format off
    explicit Network(
        int epochs = 500,
        int batch_size = 16384,
        int batches_per_epoch = 6104,
        int save_rate = 10,
        int thread_count = 2,
        float output_scalar = 400,
        float start_lambda = 0.7,
        float end_lambda = 0.8
    ) 
    : targets(batch_size) {
        // clang-format on
        Epochs = epochs;
        BatchSize = batch_size;
        BatchesPerEpoch = batches_per_epoch;
        SaveRate = save_rate;
        ThreadCount = thread_count;
        OutputScalar = output_scalar;
        StartLambda = start_lambda;
        EndLambda = end_lambda;

        create_cublas();
    }

    ~Network() {
        destroy_cublas();
    }

    void load_weights(const std::string &file) {
        std::ifstream f(file, std::ios::binary);

        // check if the file exists
        if(!f) {
            throw std::runtime_error("File " + file + " does not exist");
        }

        try {
            for(LayerBase *l : layers) {
                for(Tensor *t : l->get_params()) {
                    DenseMatrix &weights = t->get_vals();

                    f.read(reinterpret_cast<char *>(weights.host_address()), weights.size() * sizeof(float));
                    if(f.gcount() != static_cast<std::streamsize>(weights.size() * sizeof(float))) {
                        throw std::runtime_error("Error: insufficient data read from file. Expected " +
                                                 std::to_string(weights.size()) + " floats");
                    }

                    weights.host_to_dev();
                }
            }

            std::cout << "Loaded weights from " << file << std::endl;
        } catch(const std::exception &e) {
            throw std::runtime_error("Failed to load weights from " + file + ": " + e.what());
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

        DenseMatrix &output = get_output().get_vals();
        output.dev_to_host();

        return output(0) * OutputScalar;
    }

    void evaluate_positions(const std::vector<std::string> &positions) {
        std::cout << "\n================================ Testing Network ===============================\n\n";

        for(const std::string &fen : positions) {
            std::cout << "FEN: " << fen << std::endl;
            std::cout << "Eval: " << predict(fen) << std::endl;
        }
    }

    template <typename Func> //
    void set_quant_scheme(Func &&func) {
        quantFunc = std::forward<Func>(func);
    }

    void set_loss(Loss *loss) {
        ASSERT(loss != nullptr);
        this->loss = loss;
    }

    void set_optim(Optimizer *optim) {
        ASSERT(optim != nullptr);
        this->optim = optim;
    }

    void set_input_bucket(std::array<int, 64> input_bucket) {
        this->input_bucket = input_bucket;
    }

    void set_hidden_layers(std::vector<LayerBase *> hidden_layers) {
        this->layers = hidden_layers;
    }

    int get_batch_size() {
        return BatchSize;
    };

    Tensor &get_output() {
        return layers[layers.size() - 1]->get_output().activated;
    };

    std::vector<LayerBase *> get_layers() {
        return layers;
    }

    void train(std::vector<std::string> &files, std::string output_path, std::string checkpoint_name = "");
};
