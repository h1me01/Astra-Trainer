#pragma once

#include <filesystem>
#include <string>
#include <unordered_set>

#include "../../kernel/include.h"
#include "../layers/layer.h"
#include "../types.h"

namespace nn {

class Optimizer {
  public:
    Optimizer() = default;
    virtual ~Optimizer() = default;

    void init(const std::vector<LayerPtr> &layers) {
        for(const auto &l : layers) {
            for(auto &t : l->get_params()) {
                if(min_val != -1)
                    t->clamp(min_val, t->upper_bound());
                if(max_val != -1)
                    t->clamp(t->lower_bound(), max_val);
                tunables.push_back(t);
            }
        }

        init_buffers();
    }

    virtual void init_buffers() {
        for(auto *t : tunables) {
            int size = t->get_values().size();
            momentum.push_back(Array<float>{size});
            velocity.push_back(Array<float>{size});
        }
    }

    void clamp(float min, float max) {
        if(min > max)
            error("Min in optimizer cannot be greater than max");

        this->min_val = min;
        this->max_val = max;
    }

    void load(const std::string &path);
    void save(const std::string &path);

    virtual void step(float lr, int batch_size) = 0;

  protected:
    std::vector<Tensor<float> *> tunables{};

    std::vector<Array<float>> momentum{};
    std::vector<Array<float>> velocity{};

    float min_val = -1;
    float max_val = -1;
};

inline void Optimizer::load(const std::string &path) {
    std::string state_path = path + "/state";
    if(!std::filesystem::exists(state_path)) {
        error("Optimizer state path does not exist: " + state_path);
    }

    auto loadFile = [&](const std::string &filename, std::vector<Array<float>> &buffers, const std::string &name) {
        std::ifstream f(filename, std::ios::binary);
        if(!f.is_open())
            error("Failed opening file " + filename);

        for(size_t i = 0; i < buffers.size(); i++) {
            f.read(reinterpret_cast<char *>(buffers[i].host_address()), buffers[i].size() * sizeof(float));
            if(f.gcount() != static_cast<std::streamsize>(buffers[i].size() * sizeof(float))) {
                error("Insufficient data read for " + name + //
                      ". Expected " + std::to_string(buffers[i].size()) + " floats");
            }
            buffers[i].host_to_dev();
        }
    };

    try {
        loadFile(state_path + "/momentum.bin", momentum, "momentum");
        loadFile(state_path + "/velocity.bin", velocity, "velocity");
    } catch(const std::exception &e) {
        error("Failed loading optimizer state from " + state_path + ": " + e.what());
    }

    std::cout << "Loaded optimizer state from " << path << std::endl;
}

inline void Optimizer::save(const std::string &path) {
    std::string state_path = path + "/state";
    std::filesystem::create_directories(state_path);

    auto saveFile = [&](const std::string &filename, std::vector<Array<float>> &buffers) {
        std::ofstream f(filename, std::ios::binary);
        if(!f.is_open()) {
            error("Failed opening file " + filename + " for writing");
        }

        for(auto &buffer : buffers) {
            buffer.dev_to_host();
            f.write(reinterpret_cast<const char *>(buffer.host_address()), buffer.size() * sizeof(float));
            if(!f.good())
                error("Failed writing to file " + filename);
        }
    };

    try {
        saveFile(state_path + "/momentum.bin", momentum);
        saveFile(state_path + "/velocity.bin", velocity);
    } catch(const std::exception &e) {
        error("Failed saving optimizer state to " + state_path + ": " + e.what());
    }
}

} // namespace nn
