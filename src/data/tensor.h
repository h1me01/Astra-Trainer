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
        : data_(),
          grad_() {}

    Tensor(int r, int c)
        : data_(r, c),
          grad_(r, c) {
        zero_init();
    }

    void zero_init() {
        data_.clear();
        grad_.clear();
    }

    void uniform_init(float min_val, float max_val) {
        for (int i = 0; i < data_.size(); i++)
            data_(i) = std::uniform_real_distribution<float>(min_val, max_val)(rng::get_tensor_rng());
        data_.host_to_dev();
        grad_.clear();
    }

    void he_init(int input_size) {
        for (int i = 0; i < data_.size(); i++)
            data_(i) = std::normal_distribution<float>(0.0, std::sqrt(2.0 / input_size))(rng::get_tensor_rng());
        data_.host_to_dev();
        grad_.clear();
    }

    void set_bounds(float min_val, float max_val) {
        if (min_val > max_val)
            error("Tensor: Min in Tensor cannot be greater than max!");
        lower_bound_ = min_val;
        upper_bound_ = max_val;
    }

    float lower_bound() const { return lower_bound_; }
    float upper_bound() const { return upper_bound_; }

    DenseMatrix& data() { return data_; }
    const DenseMatrix& data() const { return data_; }

    DenseMatrix& grad() { return grad_; }
    const DenseMatrix& grad() const { return grad_; }

    int rows() const { return data_.rows(); }
    int cols() const { return data_.cols(); }
    int size() const { return data_.size(); }

  private:
    DenseMatrix data_;
    DenseMatrix grad_;

    float lower_bound_ = std::numeric_limits<float>::lowest();
    float upper_bound_ = std::numeric_limits<float>::max();
};

} // namespace data
