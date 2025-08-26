#pragma once

#include "../lib/dataloader.h"
#include "hyperparams.h"

class Network {
  public:
    explicit Network(const Hyperparams &hp) : hp(hp), targets(hp.batch_size) {
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
            for(auto *l : hp.layers) {
                for(auto *t : l->get_params()) {
                    auto &weights = t->get_data();

                    f.read(reinterpret_cast<char *>(weights.host_address()), weights.size() * sizeof(float));
                    if(f.gcount() != static_cast<std::streamsize>(weights.size() * sizeof(float))) {
                        error("insufficient data read from file. Expected " + //
                              std::to_string(weights.size()) + " floats");
                    }

                    weights.host_to_dev();
                }
            }

            hp.loaded_weights = file;
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

        LayerBase *output_layer = hp.layers.back();

        auto &output = get_output().get_data();
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
        this->hp.loss = loss;
    }

    void set_optim(Optimizer *optim) {
        this->hp.optim = optim;
    }

    void set_input_bucket(std::array<int, 64> input_bucket) {
        this->hp.input_bucket = input_bucket;
    }

    void set_layers(std::vector<LayerBase *> hidden_layers) {
        this->hp.layers = hidden_layers;
    }

    int get_batch_size() {
        return hp.batch_size;
    }

    Tensor<float> &get_output() {
        return hp.layers.back()->get_output().activated;
    }

    std::vector<LayerBase *> get_layers() {
        return hp.layers;
    }

    void train(std::vector<std::string> data_path, std::string output_path, std::string checkpoint_name);

  private:
    bool is_initialized = false;

    Hyperparams hp;
    Array<float> targets;

    void init() {
        if(is_initialized)
            return;

        if(hp.layers.empty())
            error("No hidden layers set for the network");
        if(hp.optim == nullptr)
            error("Optimizer is not set for the network");
        if(hp.loss == nullptr)
            error("Loss function is not set for the network");

        hp.optim->init(hp.layers);
        for(LayerBase *l : hp.layers)
            l->init(hp.batch_size);

        is_initialized = true;
    }

    void forward() {
        for(size_t i = 0; i < hp.layers.size(); i++)
            hp.layers[i]->forward();
    }

    void backward() {
        hp.loss->compute(targets, get_output());
        for(int i = hp.layers.size() - 1; i >= 0; i--)
            hp.layers[i]->backward();
    }

    void save_checkpoint(const std::string &path);

    int index(PieceType pt, Color pc, Square psq, Square ksq, Color view);

    void fill(std::vector<DataEntry> &ds, float lambda);
};
