#pragma once

#include <optional>

#include "../../data/include.h"
#include "factorizer.h"
#include "save_format.h"

namespace nn::param {

class Param {
  public:
    Param(int input_dim, int output_dim)
        : weights_(output_dim, input_dim),
          biases_(output_dim, 1) {

        if (input_dim <= 0 || output_dim <= 0)
            error("Param: Input and output dimensions must be positive!");

        weights_.he_init(input_dim);
        biases_.zero_init();
    }

    void create_factorizer(int block_size) { factorizer_ = Factorizer(&weights_, block_size); }

    bool has_factorizer() const { return factorizer_.has_value(); }

    void save(FILE* f) {
        if (has_factorizer())
            write_tensor(f, (*factorizer_).get_base());
        write_tensor(f, weights_);
        write_tensor(f, biases_);
    }

    void load(FILE* f) {
        if (has_factorizer())
            load_tensor(f, (*factorizer_).get_base());
        load_tensor(f, weights_);
        load_tensor(f, biases_);
    }

    void save_quantized(FILE* f) {
        save_tensor_quantized(f, weights_, weights_save_format_, has_factorizer());
        save_tensor_quantized(f, biases_, biases_save_format_);
    }

    void set_bounds(float min_val, float max_val) {
        weights_.set_bounds(min_val, max_val);
        biases_.set_bounds(min_val, max_val);
    }

    int get_input_dim() const { return weights_.get_data().cols(); }
    int get_output_dim() const { return weights_.get_data().rows(); }

    SaveFormat& weights_format() { return weights_save_format_; }
    SaveFormat& biases_format() { return biases_save_format_; }

    const SaveFormat& weights_format() const { return weights_save_format_; }
    const SaveFormat& biases_format() const { return biases_save_format_; }

    Tensor& get_weights() { return weights_; }
    Tensor& get_biases() { return biases_; }
    Factorizer& get_factorizer() { return *factorizer_; }

    const Tensor& get_weights() const { return weights_; }
    const Tensor& get_biases() const { return biases_; }
    const Factorizer& get_factorizer() const { return *factorizer_; }

    std::vector<Tensor*> get() {
        if (has_factorizer())
            return {&(*factorizer_).get_base(), &weights_, &biases_};
        return {&weights_, &biases_};
    }

  private:
    Tensor weights_;
    Tensor biases_;
    std::optional<Factorizer> factorizer_;
    SaveFormat weights_save_format_;
    SaveFormat biases_save_format_;

    void load_tensor(FILE* f, Tensor& tensor) {
        auto& data = tensor.get_data();
        if ((int)fread(data.host_address(), sizeof(float), data.size(), f) != data.size())
            error("Param: Failed reading tensor data from file!");
        data.host_to_dev();
    }

    void write_tensor(FILE* f, Tensor& tensor) {
        auto& data = tensor.get_data();

        data.dev_to_host();
        if ((int)fwrite(data.host_address(), sizeof(float), data.size(), f) != data.size())
            error("Param: Failed writing tensor data to file!");
    }

    template <typename T>
    void write_quantized(FILE* f, Tensor& tensor, const SaveFormat& format, bool add_factorizer = false) {
        auto& data = tensor.get_data();

        DenseMatrix facto;
        if (add_factorizer) {
            auto& facto_data = factorizer_->get_base().get_data();
            facto_data.dev_to_host();
            facto = facto_data.repeat(data.cols() / facto_data.cols());
        }

        Array<T> quantized(data.size());

        auto quantize = [&](float v) -> T {
            if constexpr (std::is_same_v<T, float>)
                return v;

            T min_qv = std::numeric_limits<T>::min();
            T max_qv = std::numeric_limits<T>::max();
            T qv = std::round(v * format.get_scale());

            if (qv < min_qv || qv > max_qv) {
                std::cout << "Warning: Value " << v << " is out of range for quantization and will be clamped!"
                          << std::endl;
            }

            return static_cast<T>(std::clamp(qv, (T)min_qv, (T)max_qv));
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
