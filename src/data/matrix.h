#pragma once

#include "array.h"

namespace data {

// column-major matrix
template <typename T>
class Matrix {
  public:
    Matrix()
        : rows_(0),
          cols_(0) {}

    Matrix(int rows, int cols)
        : rows_(rows),
          cols_(cols),
          data_(rows * cols) {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return data_.size(); }

    T operator()(int idx) const { return data_(idx); }
    T& operator()(int idx) { return data_(idx); }

    T operator()(int r, int c) const {
        CHECK(in_bounds(r, c));
        return data_.get(rows_ * c + r);
    }

    T& operator()(int r, int c) {
        CHECK(in_bounds(r, c));
        return data_.get(rows_ * c + r);
    }

    // overwrites data!
    void pinned() { data_ = Array<T>(rows_ * cols_, true); }

    void clear() { data_.clear(); }
    void clear_host() { data_.clear_host(); }
    void clear_dev() { data_.clear_dev(); }
    void host_to_dev() { data_.host_to_dev(); }
    void dev_to_host() { data_.dev_to_host(); }

    bool is_host_allocated() const { return data_.is_host_allocated(); }
    bool is_dev_allocated() const { return data_.is_dev_allocated(); }

    T* host_address() const { return data_.host_address(); }
    T* dev_address() const { return data_.dev_address(); }

    Matrix repeat(int count) const {
        Matrix result(rows_, count * cols_);
        for (int t = 0; t < count; t++)
            for (int r = 0; r < rows_; r++)
                for (int c = 0; c < cols_; c++)
                    result(r, t * cols_ + c) = (*this)(r, c);
        return result;
    }

  private:
    bool in_bounds(int r, int c) const { return r >= 0 && r < rows_ && c >= 0 && c < cols_; }

    int rows_, cols_;
    Array<T> data_;
};

using DenseMatrix = Matrix<float>;
using SparseMatrix = Matrix<int>;

} // namespace data
