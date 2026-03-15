#pragma once

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_set>

#include "../../kernel/include.h"
#include "../param/param.h"

namespace nn::optim {

using namespace param;

class Optimizer {
  public:
    Optimizer() = default;
    virtual ~Optimizer() = default;

    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;
    Optimizer(Optimizer&&) = default;
    Optimizer& operator=(Optimizer&&) = default;

    void init(const std::vector<Param*>& params) {
        for (auto* l : params)
            for (auto& t : l->get())
                this->params_.push_back(t);
        init_buffers();
    }

    void zero_grads() {
        for (auto* t : params_)
            t->get_grads().clear_dev();
    }

    void load(const std::string& path);
    void save(const std::string& path) const;

    virtual void step(float lr, int batch_size) = 0;

  protected:
    std::vector<Tensor*> params_{};

    std::vector<Array<float>> momentum_{};
    std::vector<Array<float>> velocity_{};

    virtual void init_buffers() {
        for (const auto* t : params_) {
            int size = t->get_data().size();
            momentum_.emplace_back(size);
            velocity_.emplace_back(size);
        }
    }

  private:
    void load_buffer(
        const std::filesystem::path& filepath, std::vector<Array<float>>& buffers, const std::string& buffer_name
    );

    void save_buffer(const std::filesystem::path& filepath, std::vector<Array<float>>& buffers) const;
};

inline void Optimizer::load_buffer(
    const std::filesystem::path& filepath, std::vector<Array<float>>& buffers, const std::string& buffer_name
) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        error("Optimizer: Failed to open " + buffer_name + ": " + filepath.string());
    }

    for (size_t i = 0; i < buffers.size(); ++i) {
        auto& buffer = buffers[i];
        const size_t bytes_to_read = buffer.size() * sizeof(float);

        file.read(reinterpret_cast<char*>(buffer.host_address()), bytes_to_read);

        const size_t bytes_read = file.gcount();
        if (bytes_read != bytes_to_read) {
            error(
                "Optimizer: Failed to read " + buffer_name + " buffer " + std::to_string(i) + ": expected " +
                std::to_string(bytes_to_read) + " bytes, got " + std::to_string(bytes_read) + " bytes"
            );
        }

        buffer.host_to_dev();
    }
}

inline void Optimizer::save_buffer(const std::filesystem::path& filepath, std::vector<Array<float>>& buffers) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        error("Optimizer: Failed to create state file: " + filepath.string());
    }

    for (auto& buffer : buffers) {
        buffer.dev_to_host();

        const size_t bytes_to_write = buffer.size() * sizeof(float);
        file.write(reinterpret_cast<const char*>(buffer.host_address()), bytes_to_write);

        if (!file.good()) {
            error("Optimizer: Failed to write optimizer state to: " + filepath.string());
        }
    }
}

inline void Optimizer::load(const std::string& path) {
    const std::filesystem::path state_path = std::filesystem::path(path) / "state";

    if (!std::filesystem::exists(state_path)) {
        error("Optimizer: State directory does not exist: " + state_path.string());
    }

    try {
        load_buffer(state_path / "momentum.bin", momentum_, "momentum");
        load_buffer(state_path / "velocity.bin", velocity_, "velocity");
    } catch (const std::exception& e) {
        error("Optimizer: Failed to load state from " + state_path.string() + ": " + e.what());
    }
}

inline void Optimizer::save(const std::string& path) const {
    const std::filesystem::path state_path = std::filesystem::path(path) / "state";

    try {
        std::filesystem::create_directories(state_path);

        save_buffer(state_path / "momentum.bin", const_cast<std::vector<Array<float>>&>(momentum_));
        save_buffer(state_path / "velocity.bin", const_cast<std::vector<Array<float>>&>(velocity_));
    } catch (const std::exception& e) {
        error("Optimizer: Failed to save state to " + state_path.string() + ": " + e.what());
    }
}

} // namespace nn::optim
