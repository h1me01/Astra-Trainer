#pragma once

#include "array.h"

namespace data {

// column-major dense matrix
class DenseMatrix {
  public:
    DenseMatrix()
        : rows_(0),
          cols_(0),
          data() {}

    DenseMatrix(int rows, int cols)
        : rows_(rows),
          cols_(cols),
          data(rows * cols) {}

    DenseMatrix(const DenseMatrix&) = default;
    DenseMatrix(DenseMatrix&&) noexcept = default;
    DenseMatrix& operator=(const DenseMatrix&) = default;
    DenseMatrix& operator=(DenseMatrix&&) noexcept = default;

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return data.size(); }

    float operator()(int idx) const { return data(idx); }
    float& operator()(int idx) { return data(idx); }

    void clear() { data.clear(); }
    void clear_host() { data.clear_host(); }
    void clear_dev() { data.clear_dev(); }
    void host_to_dev() { data.host_to_dev(); }
    void dev_to_host() { data.dev_to_host(); }

    DenseMatrix repeat(int times) const {
        DenseMatrix result(rows_, times * cols_);
        for (int t = 0; t < times; t++)
            for (int r = 0; r < rows_; r++)
                for (int c = 0; c < cols_; c++)
                    result(r, t * cols_ + c) = (*this)(r, c);
        return result;
    }

    bool is_host_allocated() const { return data.is_host_allocated(); }
    bool is_dev_allocated() const { return data.is_dev_allocated(); }

    float* host_address() const { return data.host_address(); }
    float* dev_address() const { return data.dev_address(); }

    float operator()(int r, int c) const {
        CHECK(r >= 0 && r < rows_ && c >= 0 && c < cols_);
        return data.get(rows_ * c + r);
    }

    float& operator()(int r, int c) {
        CHECK(r >= 0 && r < rows_ && c >= 0 && c < cols_);
        return data.get(rows_ * c + r);
    }

  private:
    int rows_, cols_;
    Array<float> data;
};

} // namespace data
