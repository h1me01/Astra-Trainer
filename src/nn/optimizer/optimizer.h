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

    void init(const std::vector<SPtr<Param>>& params) {
        for (const auto& l : params) {
            for (auto& t : l->get()) {
                if (min_val.has_value())
                    t->clamp(min_val.value(), t->upper_bound());
                if (max_val.has_value())
                    t->clamp(t->lower_bound(), max_val.value());
                this->params.push_back(t);
            }
        }

        init_buffers();
    }

    void clear_grads() {
        for (auto* t : params)
            t->get_grads().clear_dev();
    }

    void clamp(float min, float max) {
        if (min > max)
            error("Optimizer clamp: min cannot be greater than max!");

        this->min_val = min;
        this->max_val = max;
    }

    void load(const std::string& path);
    void save(const std::string& path) const;

    virtual void step(float lr, int batch_size) = 0;

  protected:
    std::vector<Tensor*> params{};

    std::vector<Array<float>> momentum{};
    std::vector<Array<float>> velocity{};

    std::optional<float> min_val;
    std::optional<float> max_val;

    virtual void init_buffers() {
        momentum.reserve(params.size());
        velocity.reserve(params.size());

        for (const auto* t : params) {
            int size = t->get_data().size();
            momentum.emplace_back(size);
            velocity.emplace_back(size);
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
        error("Failed to open optimizer " + buffer_name + " file: " + filepath.string());
    }

    for (size_t i = 0; i < buffers.size(); ++i) {
        auto& buffer = buffers[i];
        const size_t bytes_to_read = buffer.size() * sizeof(float);

        file.read(reinterpret_cast<char*>(buffer.host_address()), bytes_to_read);

        const size_t bytes_read = file.gcount();
        if (bytes_read != bytes_to_read) {
            error(
                "Failed to read " + buffer_name + " buffer " + std::to_string(i) + ": expected " +
                std::to_string(bytes_to_read) + " bytes, got " + std::to_string(bytes_read) + " bytes"
            );
        }

        buffer.host_to_dev();
    }
}

inline void Optimizer::save_buffer(const std::filesystem::path& filepath, std::vector<Array<float>>& buffers) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        error("Failed to create optimizer state file: " + filepath.string());
    }

    for (auto& buffer : buffers) {
        buffer.dev_to_host();

        const size_t bytes_to_write = buffer.size() * sizeof(float);
        file.write(reinterpret_cast<const char*>(buffer.host_address()), bytes_to_write);

        if (!file.good()) {
            error("Failed to write optimizer state to: " + filepath.string());
        }
    }
}

inline void Optimizer::load(const std::string& path) {
    const std::filesystem::path state_path = std::filesystem::path(path) / "state";

    if (!std::filesystem::exists(state_path)) {
        error("Optimizer state directory does not exist: " + state_path.string());
    }

    try {
        load_buffer(state_path / "momentum.bin", momentum, "momentum");
        load_buffer(state_path / "velocity.bin", velocity, "velocity");
    } catch (const std::exception& e) {
        error("Failed to load optimizer state from " + state_path.string() + ": " + e.what());
    }
}

inline void Optimizer::save(const std::string& path) const {
    const std::filesystem::path state_path = std::filesystem::path(path) / "state";

    try {
        std::filesystem::create_directories(state_path);

        save_buffer(state_path / "momentum.bin", const_cast<std::vector<Array<float>>&>(momentum));
        save_buffer(state_path / "velocity.bin", const_cast<std::vector<Array<float>>&>(velocity));
    } catch (const std::exception& e) {
        error("Failed to save optimizer state to " + state_path.string() + ": " + e.what());
    }
}

} // namespace nn::optim
