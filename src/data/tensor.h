#pragma once

#include <algorithm>
#include <iostream>
#include <random>

#include "matrix.h"

namespace data {

namespace rng {

inline auto& get_tensor_rng() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

inline void reset_tensor_rng() {
    get_tensor_rng().seed(42);
}

} // namespace rng

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

    void zero_init() {
        data.clear();
        grads.clear();
    }

    void uniform_init(float min_val, float max_val) {
        for (int i = 0; i < data.size(); i++)
            data(i) = std::uniform_real_distribution<float>(min_val, max_val)(rng::get_tensor_rng());
        data.host_to_dev();
        grads.clear();
    }

    void he_init(int input_size) {
        for (int i = 0; i < data.size(); i++)
            data(i) = std::normal_distribution<float>(0.0, std::sqrt(2.0 / input_size))(rng::get_tensor_rng());
        data.host_to_dev();
        grads.clear();
    }

    void set_bounds(float min_val, float max_val) {
        if (min_val > max_val)
            error("Tensor: Min in Tensor cannot be greater than max!");
        lower_bound_ = min_val;
        upper_bound_ = max_val;
    }

    float lower_bound() const { return lower_bound_; }
    float upper_bound() const { return upper_bound_; }

    DenseMatrix& get_data() { return data; }
    const DenseMatrix& get_data() const { return data; }

    DenseMatrix& get_grads() { return grads; }
    const DenseMatrix& get_grads() const { return grads; }

    int rows() const { return data.rows(); }
    int cols() const { return data.cols(); }
    int size() const { return data.size(); }

  private:
    DenseMatrix data;
    DenseMatrix grads;

    float lower_bound_ = std::numeric_limits<float>::lowest();
    float upper_bound_ = std::numeric_limits<float>::max();
};

} // namespace data
