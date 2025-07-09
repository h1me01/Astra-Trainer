#pragma once

#include <algorithm>
#include <iostream>
#include <random>

#include "dense_matrix.h"

enum class WeightInitType { Uniform, He, Xavier };

enum class QuantType { //
    INT8,
    INT16,
    INT32,
    FLOAT
};

struct QuantScheme {
    int scale = 1;
    bool trans = false;
    QuantType type = QuantType::FLOAT;

    QuantScheme() {};
};

class Tensor {
  private:
    DenseMatrix<float> data;
    DenseMatrix<float> grads;

    float m_lower_bound = std::numeric_limits<float>::lowest();
    float m_upper_bound = std::numeric_limits<float>::max();

    QuantScheme quant_scheme;

    template <typename T> //
    void write_quantized(FILE *f);

  public:
    Tensor(int r, int c)                  //
        : data(DenseMatrix<float>{r, c}), //
          grads(DenseMatrix<float>{r, c}) //
    {
        data.clear();
        grads.clear();
    }

    Tensor(const Tensor &other)
        : data(other.data),                   //
          grads(other.grads),                 //
          m_lower_bound(other.m_lower_bound), //
          m_upper_bound(other.m_upper_bound), //
          quant_scheme(other.quant_scheme) {}

    Tensor &operator=(const Tensor &other) {
        if(this != &other) {
            data = other.data;
            grads = other.grads;
            m_lower_bound = other.m_lower_bound;
            m_upper_bound = other.m_upper_bound;
            quant_scheme = other.quant_scheme;
        }
        return *this;
    }

    void init(WeightInitType type, int previous_size = 0);

    template <QuantType type> //
    void quantize(int scale, bool transpose = false) {
        if(scale <= 0)
            error("Quantize scale must be greater than 0");

        quant_scheme.scale = scale;
        quant_scheme.type = type;
        quant_scheme.trans = transpose;
    }

    void save_quantize(FILE *f) {
        switch(quant_scheme.type) {
        case QuantType::INT8:
            write_quantized<int8_t>(f);
            break;
        case QuantType::INT16:
            write_quantized<int16_t>(f);
            break;
        case QuantType::INT32:
            write_quantized<int32_t>(f);
            break;
        case QuantType::FLOAT:
            write_quantized<float>(f);
            break;
        default:
            error("Unknown quantization type");
        }
    }

    void clamp(float min_val, float max_val) {
        if(min_val > max_val)
            error("Min in Tensor cannot be greater than max");

        this->m_lower_bound = min_val;
        this->m_upper_bound = max_val;
    }

    float lower_bound() const {
        return m_lower_bound;
    }

    float upper_bound() const {
        return m_upper_bound;
    }

    DenseMatrix<float> &get_data() {
        return data;
    }

    DenseMatrix<float> &get_grads() {
        return grads;
    }
};

inline void Tensor::init(WeightInitType type, int previous_size) {
    std::mt19937 gen{std::random_device{}()};

    auto fill_data = [&](auto distribution) {
        for(int i = 0; i < data.size(); i++)
            data(i) = distribution(gen);
    };

    switch(type) {
    case WeightInitType::Uniform:
        fill_data(std::uniform_real_distribution<>(-0.1f, 0.1f));
        break;
    case WeightInitType::He:
        fill_data(std::normal_distribution<>(0.0f, std::sqrt(2.0f / previous_size)));
        break;
    case WeightInitType::Xavier:
        fill_data(std::normal_distribution<>(0.0f, std::sqrt(1.0f / previous_size)));
        break;
    default:
        error("Unknown weight initialization type");
    }

    data.host_to_dev();
    grads.clear();
}

template <typename T> //
void Tensor::write_quantized(FILE *f) {
    auto quantize_value = [&](float orig) -> T {
        if constexpr(std::is_same_v<T, float>) {
            return orig; // no quantization needed for float
        } else {
            float scaled = orig * quant_scheme.scale;
            T quant = static_cast<T>(round(scaled));

            if(quant < std::numeric_limits<T>::min() || quant > std::numeric_limits<T>::max()) {
                std::stringstream ss;
                ss << "Overflow/Underflow while quantizing value: " << orig;
                ss << " with scale: " << quant_scheme.scale << ". ";
                ss << "Quantized value: " << static_cast<long long>(quant) << ". ";
                ss << "Type: " << typeid(T).name() << ".";
                error(ss.str());
            }
            return quant;
        }
    };

    Array<T> quantized = Array<T>(data.size());

    if(quant_scheme.trans) {
        const int rows = data.rows();
        const int cols = data.cols();

        for(int r = 0; r < rows; r++)
            for(int c = 0; c < cols; c++)
                quantized(cols * r + c) = quantize_value(data(r, c));
    } else {
        for(int i = 0; i < data.size(); i++)
            quantized(i) = quantize_value(data(i));
    }

    int written = fwrite(quantized.host_address(), sizeof(T), quantized.size(), f);
    if(written != quantized.size()) {
        std::stringstream ss;
        ss << "Failed writing quantized data to file! ";
        ss << "Expected " << quantized.size() << " elements, but wrote " << written << " elements";
        error(ss.str());
    }
}
