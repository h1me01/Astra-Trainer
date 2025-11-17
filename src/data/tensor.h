#pragma once

#include <algorithm>
#include <iostream>
#include <random>

#include "dense_matrix.h"

namespace data {

enum class WeightInitType { Uniform, He, Xavier };

enum class QuantType { INT8, INT16, FLOAT };

class Tensor {
  public:
    Tensor() : values(), gradients() {}

    Tensor(int r, int c) : values(r, c), gradients(r, c) {
        values.clear();
        gradients.clear();
    }

    Tensor(const Tensor &) = default;
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    void init(WeightInitType type, int input_size = 0) {
        std::mt19937 gen{std::random_device{}()};

        for(int i = 0; i < values.size(); i++) {
            switch(type) {
            case WeightInitType::Uniform:
                values(i) = std::uniform_real_distribution<float>(-0.1, 0.1)(gen);
                break;
            case WeightInitType::He:
                values(i) = std::normal_distribution<float>(0.0, std::sqrt(2.0 / input_size))(gen);
                break;
            case WeightInitType::Xavier:
                values(i) = std::normal_distribution<float>(0.0, std::sqrt(1.0 / input_size))(gen);
                break;
            }
        }

        values.host_to_dev();
        gradients.clear();
    }

    void load(FILE *f) {
        if(fread(values.host_address(), sizeof(float), values.size(), f) != values.size())
            error("Failed reading tensor data from file!");
        values.host_to_dev();
    }

    void save(FILE *f) {
        values.dev_to_host();
        if(fwrite(values.host_address(), sizeof(float), values.size(), f) != values.size())
            error("Failed writing tensor data to file!");
    }

    void save_quantized(FILE *f) {
        values.dev_to_host();

        switch(m_quant_type) {
        case QuantType::INT8:
            write_quantized<int8_t>(f);
            break;
        case QuantType::INT16:
            write_quantized<int16_t>(f);
            break;
        case QuantType::FLOAT:
            write_quantized<float>(f);
            break;
        default:
            error("Unknown quantization type!");
        }
    }

    void clamp(float min_val, float max_val) {
        if(min_val > max_val)
            error("Min in Tensor cannot be greater than max!");
        m_lower_bound = min_val;
        m_upper_bound = max_val;
    }

    // clang-format off
    float lower_bound() const { return m_lower_bound; }
    float upper_bound() const { return m_upper_bound; }

    DenseMatrix& get_values() { return values; }
    const DenseMatrix& get_values() const { return values; }
    
    DenseMatrix& get_gradients() { return gradients; }
    const DenseMatrix& get_gradients() const { return gradients; }
    // clang-format on

    Tensor &quant_scale(int scale) {
        m_quant_scale = scale;
        return *this;
    }

    Tensor &quant_type(QuantType type) {
        m_quant_type = type;
        return *this;
    }

    // only transposes weights when quantizing
    Tensor &transpose() {
        m_transpose = true;
        return *this;
    }

  private:
    DenseMatrix values;
    DenseMatrix gradients;

    int m_quant_scale = 1;
    bool m_transpose = false;
    QuantType m_quant_type = QuantType::FLOAT;

    float m_lower_bound = std::numeric_limits<float>::lowest();
    float m_upper_bound = std::numeric_limits<float>::max();

    template <typename T> //
    void write_quantized(FILE *f) {
        Array<T> quantized(values.size());

        auto quantize = [&](float v) -> T {
            if constexpr(std::is_same_v<T, float>)
                return v;
            return static_cast<T>(std::clamp( //
                std::round(v * m_quant_scale),
                (float) std::numeric_limits<T>::min(),
                (float) std::numeric_limits<T>::max()));
        };

        if(m_transpose) {
            for(int r = 0; r < values.rows(); r++)
                for(int c = 0; c < values.cols(); c++)
                    quantized(values.cols() * r + c) = quantize(values(r, c));
        } else {
            for(int i = 0; i < values.size(); i++)
                quantized(i) = quantize(values(i));
        }

        if(fwrite(quantized.host_address(), sizeof(T), quantized.size(), f) != quantized.size())
            error("Failed writing quantized data to file!");
    }
};

} // namespace data
