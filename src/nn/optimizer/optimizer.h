#pragma once

#include <filesystem>
#include <string>

#include "../../kernel/include.h"
#include "../layers/layer.h"
#include "../lr_scheduler.h"

class Optimizer {
  protected:
    std::string name = "Optimizer";

    std::vector<Tensor *> tunables{};

    std::vector<Array<float>> momentum{};
    std::vector<Array<float>> velocity{};
    std::vector<Array<float>> slow_buffer{}; // used by ranger

    int step = 0;

    AdamParams params;

    float min_val = -1;
    float max_val = -1;

    LRScheduler *lr_scheduler = nullptr;

  public:
    Optimizer(AdamParams params) //
        : params(params) {}

    void init(std::vector<LayerBase *> layers) {
        for(LayerBase *l : layers) {
            for(auto *t : l->get_params()) {
                if(min_val != -1)
                    t->clamp(min_val, t->upper_bound());
                if(max_val != -1)
                    t->clamp(t->lower_bound(), max_val);
                tunables.push_back(t);
            }
        }
        init_buffers();
    }

    void lr_from_epoch(int epoch) {
        if(lr_scheduler == nullptr)
            return;
        for(int i = 1; i <= epoch; i++)
            update_lr(i);
    }

    void update_lr(int epoch) {
        if(lr_scheduler != nullptr)
            params.lr = lr_scheduler->get_lr(epoch, params.lr);
    }

    void clamp(float min, float max) {
        if(min > max)
            error("Error: min cannot be greater than max.");

        this->min_val = min;
        this->max_val = max;
    }

    void set_lr_scheduler(LRScheduler *scheduler) {
        this->lr_scheduler = scheduler;
    }

    float get_lr() const {
        return params.lr;
    }

    void load(const std::string &path);
    void save(const std::string &path);

    virtual void apply(int batch_size) = 0;

    void init_buffers() {
        for(Tensor *t : tunables) {
            int size = t->get_data().size();
            momentum.push_back(Array<float>{size});
            velocity.push_back(Array<float>{size});

            if(name == "Ranger")
                slow_buffer.push_back(Array<float>{size});
        }
    }

    std::string get_info() {
        std::stringstream ss;
        ss << name << "(";
        ss << "lr=" << format_number(params.lr);
        ss << ", beta1=" << format_number(params.beta1);
        ss << ", beta2=" << format_number(params.beta2);
        ss << ", eps=" << format_number(params.eps);
        if(params.decay != 0.0f)
            ss << ", decay=" << format_number(params.decay);
        if(lr_scheduler != nullptr)
            ss << ", lr_scheduler=" << lr_scheduler->get_info();
        ss << ")";
        return ss.str();
    }
};

inline void Optimizer::load(const std::string &path) {
    std::string state_path = path + "/state";
    if(!std::filesystem::exists(state_path)) {
        error("Optimizer state path does not exist: " + state_path);
    }

    auto loadFile = [&](const std::string &filename, std::vector<Array<float>> &buffers, const std::string &name) {
        std::ifstream f(filename, std::ios::binary);
        if(!f.is_open())
            error("Failed to open file " + filename);

        for(size_t i = 0; i < buffers.size(); i++) {
            f.read(reinterpret_cast<char *>(buffers[i].host_address()), buffers[i].size() * sizeof(float));
            if(f.gcount() != static_cast<std::streamsize>(buffers[i].size() * sizeof(float))) {
                error("Error: insufficient data read for " + name + //
                      ". Expected " + std::to_string(buffers[i].size()) + " floats");
            }
            buffers[i].host_to_dev();
        }
    };

    try {
        loadFile(state_path + "/momentum.bin", momentum, "momentum");
        loadFile(state_path + "/velocity.bin", velocity, "velocity");
        if(name == "Ranger")
            loadFile(state_path + "/slow_buffer.bin", slow_buffer, "slow_buffer");
    } catch(const std::exception &e) {
        error("Failed to load optimizer state from " + state_path + ": " + e.what());
    }

    std::cout << "Loaded optimizer state from " << path << std::endl;
}

inline void Optimizer::save(const std::string &path) {
    std::string state_path = path + "/state";
    std::filesystem::create_directories(state_path);

    auto saveFile = [&](const std::string &filename, std::vector<Array<float>> &buffers) {
        std::ofstream f(filename, std::ios::binary);
        if(!f.is_open()) {
            error("Failed to open file " + filename + " for writing");
        }

        for(auto &buffer : buffers) {
            buffer.dev_to_host();
            f.write(reinterpret_cast<const char *>(buffer.host_address()), buffer.size() * sizeof(float));
            if(!f.good())
                error("Error writing to file " + filename);
        }
    };

    try {
        saveFile(state_path + "/momentum.bin", momentum);
        saveFile(state_path + "/velocity.bin", velocity);
        if(name == "Ranger")
            saveFile(state_path + "/slow_buffer.bin", slow_buffer);
    } catch(const std::exception &e) {
        error("Failed to save optimizer state to " + state_path + ": " + e.what());
    }
}
