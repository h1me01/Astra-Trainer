#pragma once

#include <algorithm>
#include <iostream>
#include <random>

#include "dense_matrix.h"

namespace data {

enum class WeightInitType { Uniform, He, Xavier };

enum class QuantType { INT8, INT16, INT32, FLOAT };

template <typename T> //
class Tensor {
  private:
    struct QuantScheme {
        int scale = 1;
        bool trans = false;
        QuantType type = QuantType::FLOAT;
    };

  public:
    Tensor(int r, int c) : values(r, c), gradients(r, c) {
        values.clear();
        gradients.clear();
    }

    Tensor(const Tensor &) = default;
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    void init(WeightInitType type, int input_size) {
        std::mt19937 gen{std::random_device{}()};

        auto fill_data = [&](auto distribution) {
            for(int i = 0; i < values.size(); i++)
                values(i) = distribution(gen);
        };

        switch(type) {
        case WeightInitType::Uniform:
            fill_data(std::uniform_real_distribution<T>(-0.1, 0.1));
            break;
        case WeightInitType::He:
            ASSERT(input_size > 0);
            fill_data(std::normal_distribution<T>(0.0, std::sqrt(2.0 / input_size)));
            break;
        case WeightInitType::Xavier:
            ASSERT(input_size > 0);
            fill_data(std::normal_distribution<T>(0.0, std::sqrt(1.0 / input_size)));
            break;
        default:
            error("Unknown weight initialization type");
        }

        values.host_to_dev();
        gradients.clear();
    }

    void quantize(QuantType type, int scale, bool transpose = false) {
        if(scale <= 0)
            error("Quantize scale must be greater than 0!");

        quant_scheme.scale = scale;
        quant_scheme.type = type;
        quant_scheme.trans = transpose;
    }

    void save_quantized(FILE *f) {
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

    void clamp(T min_val, T max_val) {
        if(min_val > max_val)
            error("Min in Tensor cannot be greater than max!");

        m_lower_bound = min_val;
        m_upper_bound = max_val;
    }

    // clang-format off
    T lower_bound() const { return m_lower_bound; }
    T upper_bound() const { return m_upper_bound; }

    DenseMatrix<T>& get_values() { return values; }
    const DenseMatrix<T>& get_values() const { return values; }
    
    DenseMatrix<T>& get_gradients() { return gradients; }
    const DenseMatrix<T>& get_gradients() const { return gradients; }
    // clang-format on

  private:
    DenseMatrix<T> values;
    DenseMatrix<T> gradients;

    QuantScheme quant_scheme;

    T m_lower_bound = std::numeric_limits<T>::lowest();
    T m_upper_bound = std::numeric_limits<T>::max();

    template <typename QT> //
    void write_quantized(FILE *f) {
        auto quantize_value = [&](T orig) -> QT {
            if constexpr(std::is_same_v<QT, T>) {
                return orig;
            } else {
                T scaled = orig * quant_scheme.scale;
                QT quant = static_cast<QT>(std::round(scaled));

                constexpr auto qmin = std::numeric_limits<QT>::min();
                constexpr auto qmax = std::numeric_limits<QT>::max();

                if(quant < qmin || quant > qmax) {
                    std::stringstream ss;
                    ss << "Quantization overflow: " << orig << " * " << quant_scheme.scale << " = " //
                       << (long long) quant << " (range: [" << (long long) qmin << ", "             //
                       << (long long) qmax << "])";
                    error(ss.str());
                }

                return quant;
            }
        };

        Array<QT> quantized(values.size());

        if(quant_scheme.trans) {
            const int rows = values.rows();
            const int cols = values.cols();

            for(int r = 0; r < rows; r++)
                for(int c = 0; c < cols; c++)
                    quantized(cols * r + c) = quantize_value(values(r, c));
        } else {
            for(int i = 0; i < values.size(); i++)
                quantized(i) = quantize_value(values(i));
        }

        size_t written = std::fwrite(quantized.host_address(), sizeof(QT), quantized.size(), f);
        if(written != static_cast<size_t>(quantized.size())) {
            std::stringstream ss;
            ss << "Failed writing quantized data to file! Expected " << quantized.size() << " elements, but wrote "
               << written;
            error(ss.str());
        }
    }
};

} // namespace data
