#pragma once

#include "../../data/include.h"

namespace nn::param {

class SaveFormat {
  public:
    enum class Type { int8, int16, float32 };

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
    Type m_type = Type::float32;
};

class Param {
  public:
    Param(int input_dim, int output_dim)
        : weights(output_dim, input_dim),
          biases(output_dim, 1) {

        if (input_dim <= 0 || output_dim <= 0)
            error("Param: Input and output dimensions must be positive!");

        weights.he_init(input_dim);
        biases.zero_init();
    }

    void create_factorizer() { factorizer_weights = Tensor(weights.get_data().rows(), 768); }

    void save(FILE* f) {
        if (factorizer_weights.size() > 0)
            write_tensor(f, factorizer_weights);
        write_tensor(f, weights);
        write_tensor(f, biases);
    }

    void load(FILE* f) {
        if (factorizer_weights.size() > 0)
            load_tensor(f, factorizer_weights);
        load_tensor(f, weights);
        load_tensor(f, biases);
    }

    void save_quantized(FILE* f) {
        save_tensor_quantized(f, weights, weights_save_format, factorizer_weights.size() > 0);
        save_tensor_quantized(f, biases, biases_save_format);
    }

    int get_input_dim() const { return weights.get_data().cols(); }
    int get_output_dim() const { return weights.get_data().rows(); }

    SaveFormat& weights_format() { return weights_save_format; }
    SaveFormat& biases_format() { return biases_save_format; }

    const SaveFormat& weights_format() const { return weights_save_format; }
    const SaveFormat& biases_format() const { return biases_save_format; }

    Tensor& get_weights() { return weights; }
    Tensor& get_biases() { return biases; }
    Tensor& get_factorizer_weights() { return factorizer_weights; }

    const Tensor& get_weights() const { return weights; }
    const Tensor& get_biases() const { return biases; }
    const Tensor& get_factorizer_weights() const { return factorizer_weights; }

    std::vector<Tensor*> get() {
        if (factorizer_weights.size() > 0)
            return {&factorizer_weights, &weights, &biases};
        return {&weights, &biases};
    }

  private:
    Tensor weights;
    Tensor biases;
    Tensor factorizer_weights; // currently a very ugly solution so TODO refactor
    SaveFormat weights_save_format;
    SaveFormat biases_save_format;

    void load_tensor(FILE* f, Tensor& tensor) {
        auto& data = tensor.get_data();
        if ((int)fread(data.host_address(), sizeof(float), data.size(), f) != data.size())
            error("Failed reading tensor data from file!");
        data.host_to_dev();
    }

    void write_tensor(FILE* f, Tensor& tensor) {
        auto& data = tensor.get_data();

        data.dev_to_host();
        if ((int)fwrite(data.host_address(), sizeof(float), data.size(), f) != data.size())
            error("Failed writing tensor data to file!");
    }

    template <typename T>
    void write_quantized(FILE* f, Tensor& tensor, const SaveFormat& format, bool add_factorizer = false) {
        auto& data = tensor.get_data();

        DenseMatrix facto;
        if (add_factorizer) {
            factorizer_weights.get_data().dev_to_host();

            // repeat to match weights dimensions
            auto& facto_data = factorizer_weights.get_data();
            const int num_repeats = data.cols() / facto_data.cols();

            facto = DenseMatrix(data.rows(), data.cols());

            for (int r = 0; r < data.rows(); r++)
                for (int rep = 0; rep < num_repeats; rep++)
                    for (int c = 0; c < facto_data.cols(); c++)
                        facto(r, rep * facto_data.cols() + c) = facto_data(r, c);
        }

        Array<T> quantized(data.size());

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
            for (int r = 0; r < data.rows(); r++)
                for (int c = 0; c < data.cols(); c++)
                    quantized(data.cols() * r + c) = quantize(data(r, c) + (add_factorizer ? facto(r, c) : 0.0f));
        } else {
            for (int i = 0; i < data.size(); i++)
                quantized(i) = quantize(data(i) + (add_factorizer ? facto(i) : 0.0f));
        }

        if ((int)fwrite(quantized.host_address(), sizeof(T), quantized.size(), f) != quantized.size())
            error("Failed writing quantized data to file!");
    }

    void save_tensor_quantized(FILE* f, Tensor& tensor, const SaveFormat& format, bool add_factorizer = false) {
        tensor.get_data().dev_to_host();
        switch (format.get_type()) {
        case SaveFormat::Type::int8:
            write_quantized<int8_t>(f, tensor, format, add_factorizer);
            break;
        case SaveFormat::Type::int16:
            write_quantized<int16_t>(f, tensor, format, add_factorizer);
            break;
        case SaveFormat::Type::float32:
            write_quantized<float>(f, tensor, format, add_factorizer);
            break;
        }
    }
};

} // namespace nn::param
