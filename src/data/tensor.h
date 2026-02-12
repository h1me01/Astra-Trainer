#pragma once

#include <algorithm>
#include <iostream>
#include <random>

#include "dense_matrix.h"

namespace data {

class Tensor {
  public:
    Tensor()
        : data(),
          grads() {}

    Tensor(int r, int c)
        : data(r, c),
          grads(r, c) {
        zero_init();
    }

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    void zero_init() {
        data.clear();
        grads.clear();
    }

    void uniform_init(float min_val, float max_val) {
        std::mt19937 gen{std::random_device{}()};
        for (int i = 0; i < data.size(); i++)
            data(i) = std::uniform_real_distribution<float>(min_val, max_val)(gen);
        data.host_to_dev();
        grads.clear();
    }

    void he_init(int input_size) {
        std::mt19937 gen{std::random_device{}()};
        for (int i = 0; i < data.size(); i++)
            data(i) = std::normal_distribution<float>(0.0, std::sqrt(2.0 / input_size))(gen);
        data.host_to_dev();
        grads.clear();
    }

    void clamp(float min_val, float max_val) {
        if (min_val > max_val)
            error("Min in Tensor cannot be greater than max!");
        m_lower_bound = min_val;
        m_upper_bound = max_val;
    }

    float lower_bound() const { return m_lower_bound; }
    float upper_bound() const { return m_upper_bound; }

    DenseMatrix& get_data() { return data; }
    const DenseMatrix& get_data() const { return data; }

    DenseMatrix& get_grads() { return grads; }
    const DenseMatrix& get_grads() const { return grads; }

    void free() {
        data.free();
        grads.free();
    }

  private:
    DenseMatrix data;
    DenseMatrix grads;

    float m_lower_bound = std::numeric_limits<float>::lowest();
    float m_upper_bound = std::numeric_limits<float>::max();
};

} // namespace data
