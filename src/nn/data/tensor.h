#pragma once

#include <algorithm>
#include <iostream>
#include <random>

#include "dense_matrix.h"

enum class WeightInitType { Uniform, He };

enum class QuantType { INT8, INT16, INT32 };

struct QuantScheme {
    int scale = 64;
    bool trans = false;
    QuantType type = QuantType::INT16;

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
    void write_quantized(FILE *f) {
        auto quantize_value = [&](float orig) {
            T quant = static_cast<T>(round(orig * quant_scheme.scale));
            if(quant < std::numeric_limits<T>::min() || quant > std::numeric_limits<T>::max()) {
                std::cout << "Overflow/Underflow while quantizing: quant = " << static_cast<int>(quant)
                          << " | orig = " << orig << "\n";
                exit(1);
            }
            return quant;
        };

        Array<T> quantized = Array<T>(data.size());

        if(quant_scheme.trans) {
            int idx = 0;
            for(int r = 0; r < data.num_rows(); r++)
                for(int c = 0; c < data.num_cols(); c++)
                    quantized(idx++) = quantize_value(data(c, r));
        } else {
            for(int i = 0; i < data.size(); i++)
                quantized(i) = quantize_value(data(i));
        }

        int written = fwrite(quantized.host_address(), sizeof(T), quantized.size(), f);
        if(written != quantized.size()) {
            std::cerr << "Error writing quantized data to file. Expected " << quantized.size()
                      << " elements, but wrote " << written << " elements." << std::endl;
            exit(1);
        }
    }

  public:
    Tensor(int r, int c)                  //
        : data(DenseMatrix<float>{r, c}), //
          grads(DenseMatrix<float>{r, c}) //
    {
        data.clear();
        grads.clear();
    }

    Tensor(const Tensor &other) //
        : data(other.data), grads(other.grads) {}

    Tensor &operator=(const Tensor &other) {
        if(this != &other) {
            data = other.data;
            grads = other.grads;
        }
        return *this;
    }

    void init(WeightInitType type, int previous_size = 0) {
        std::mt19937 gen{std::random_device{}()};

        std::uniform_real_distribution<> dis;
        if(type == WeightInitType::Uniform)
            dis = std::uniform_real_distribution<>(-0.1f, 0.1f);
        else if(type == WeightInitType::He)
            dis = std::uniform_real_distribution<>(0, std::sqrt(2.0f / previous_size));
        else {
            std::cerr << "Unknown weight initialization type!" << std::endl;
            exit(1);
        }

        for(int i = 0; i < data.size(); i++)
            data(i) = dis(gen);
        data.host_to_dev();

        grads.clear();
    }

    template <QuantType type> //
    void quantize(int scale, bool transpose = false) {
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
        default:
            std::cerr << "Unknown quantization type!" << std::endl;
            exit(1);
        }
    }

    void clamp(float min_val, float max_val) {
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