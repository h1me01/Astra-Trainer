#pragma once

#include <algorithm>
#include <iostream>
#include <random>

#include "dense_matrix.h"

namespace data {

class Tensor {
  public:
    Tensor()
        : values(),
          gradients() {}

    Tensor(int r, int c)
        : values(r, c),
          gradients(r, c) {
        zero_init();
    }

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    void zero_init() {
        values.clear();
        gradients.clear();
    }

    void uniform_init(float min_val, float max_val) {
        std::mt19937 gen{std::random_device{}()};
        for (int i = 0; i < values.size(); i++)
            values(i) = std::uniform_real_distribution<float>(min_val, max_val)(gen);
        values.host_to_dev();
        gradients.clear();
    }

    void he_init(int input_size) {
        std::mt19937 gen{std::random_device{}()};
        for (int i = 0; i < values.size(); i++)
            values(i) = std::normal_distribution<float>(0.0, std::sqrt(2.0 / input_size))(gen);
        values.host_to_dev();
        gradients.clear();
    }

    void clamp(float min_val, float max_val) {
        if (min_val > max_val)
            error("Min in Tensor cannot be greater than max!");
        m_lower_bound = min_val;
        m_upper_bound = max_val;
    }

    float lower_bound() const { return m_lower_bound; }
    float upper_bound() const { return m_upper_bound; }

    DenseMatrix& get_values() { return values; }
    const DenseMatrix& get_values() const { return values; }

    DenseMatrix& get_gradients() { return gradients; }
    const DenseMatrix& get_gradients() const { return gradients; }

  private:
    DenseMatrix values;
    DenseMatrix gradients;

    float m_lower_bound = std::numeric_limits<float>::lowest();
    float m_upper_bound = std::numeric_limits<float>::max();
};

} // namespace data
