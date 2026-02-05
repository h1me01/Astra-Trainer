#pragma once

#include "../../data/include.h"

namespace nn {

class SaveFormat {
  public:
    enum class Type { INT8, INT16, FLOAT };

    SaveFormat& transpose() {
        m_transpose = true;
        return *this;
    }

    SaveFormat& type(Type t) {
        m_type = t;
        return *this;
    }

    SaveFormat& scale(int s) {
        m_scale = s;
        return *this;
    }

    Type get_type() const { return m_type; }
    int get_scale() const { return m_scale; }
    bool is_transposed() const { return m_transpose; }

  private:
    int m_scale = 1;
    bool m_transpose = false;
    Type m_type = Type::FLOAT;
};

class Params {
  public:
    Params(int input_dim, int output_dim)
        : weights(output_dim, input_dim),
          biases(output_dim, 1) {
        weights.he_init(input_dim);
        biases.zero_init();
    }

    void save(FILE* f) {
        write_tensor(f, weights);
        write_tensor(f, biases);
    }

    void load(FILE* f) {
        load_tensor(f, weights);
        load_tensor(f, biases);
    }

    void save_quantized(FILE* f) {
        save_tensor_quantized(f, weights, weights_save_format);
        save_tensor_quantized(f, biases, biases_save_format);
    }

    int get_input_dim() const { return weights.get_values().cols(); }
    int get_output_dim() const { return weights.get_values().rows(); }

    SaveFormat& weights_format() { return weights_save_format; }
    SaveFormat& biases_format() { return biases_save_format; }

    const SaveFormat& weights_format() const { return weights_save_format; }
    const SaveFormat& biases_format() const { return biases_save_format; }

    Tensor& get_weights() { return weights; }
    Tensor& get_biases() { return biases; }

    const Tensor& get_weights() const { return weights; }
    const Tensor& get_biases() const { return biases; }

    std::vector<Tensor*> get() { return {&weights, &biases}; }

  private:
    Tensor weights;
    Tensor biases;
    SaveFormat weights_save_format;
    SaveFormat biases_save_format;

    void load_tensor(FILE* f, Tensor& tensor) {
        auto& values = tensor.get_values();
        if ((int)fread(values.host_address(), sizeof(float), values.size(), f) != values.size())
            error("Failed reading tensor data from file!");
        values.host_to_dev();
    }

    void write_tensor(FILE* f, Tensor& tensor) {
        auto& value = tensor.get_values();

        value.dev_to_host();
        if ((int)fwrite(value.host_address(), sizeof(float), value.size(), f) != value.size())
            error("Failed writing tensor data to file!");
    }

    template <typename T>
    void write_quantized(FILE* f, Tensor& tensor, const SaveFormat& format) {
        auto& values = tensor.get_values();
        Array<T> quantized(values.size());

        auto quantize = [&](float v) -> T {
            if constexpr (std::is_same_v<T, float>)
                return v;
            return static_cast<T>(std::clamp(
                std::round(v * format.get_scale()),
                (float)std::numeric_limits<T>::min(),
                (float)std::numeric_limits<T>::max()
            ));
        };

        if (format.is_transposed()) {
            for (int r = 0; r < values.rows(); r++)
                for (int c = 0; c < values.cols(); c++)
                    quantized(values.cols() * r + c) = quantize(values(r, c));
        } else {
            for (int i = 0; i < values.size(); i++)
                quantized(i) = quantize(values(i));
        }

        if ((int)fwrite(quantized.host_address(), sizeof(T), quantized.size(), f) != quantized.size())
            error("Failed writing quantized data to file!");
    }

    void save_tensor_quantized(FILE* f, Tensor& tensor, const SaveFormat& format) {
        tensor.get_values().dev_to_host();
        switch (format.get_type()) {
        case SaveFormat::Type::INT8:
            write_quantized<int8_t>(f, tensor, format);
            break;
        case SaveFormat::Type::INT16:
            write_quantized<int16_t>(f, tensor, format);
            break;
        case SaveFormat::Type::FLOAT:
            write_quantized<float>(f, tensor, format);
            break;
        }
    }
};

} // namespace nn
