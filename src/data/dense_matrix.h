#pragma once

#include "array.h"

namespace data {

// column-major dense matrix
class DenseMatrix {
  public:
    DenseMatrix()
        : m_rows(0),
          m_cols(0),
          data() {}

    DenseMatrix(int rows, int cols)
        : m_rows(rows),
          m_cols(cols),
          data(rows * cols) {}

    DenseMatrix(const DenseMatrix&) = default;
    DenseMatrix(DenseMatrix&&) noexcept = default;
    DenseMatrix& operator=(const DenseMatrix&) = default;
    DenseMatrix& operator=(DenseMatrix&&) noexcept = default;

    int rows() const { return m_rows; }
    int cols() const { return m_cols; }
    int size() const { return data.size(); }

    float operator()(int idx) const { return data(idx); }
    float& operator()(int idx) { return data(idx); }

    void clear() { data.clear(); }
    void clear_host() { data.clear_host(); }
    void clear_dev() { data.clear_dev(); }
    void host_to_dev() { data.host_to_dev(); }
    void dev_to_host() { data.dev_to_host(); }

    bool is_host_allocated() const { return data.is_host_allocated(); }
    bool is_dev_allocated() const { return data.is_dev_allocated(); }

    float* host_address() const { return data.host_address(); }
    float* dev_address() const { return data.dev_address(); }

    float operator()(int r, int c) const {
        ASSERT(r >= 0 && r < m_rows && c >= 0 && c < m_cols);
        return data.get(m_rows * c + r);
    }

    float& operator()(int r, int c) {
        ASSERT(r >= 0 && r < m_rows && c >= 0 && c < m_cols);
        return data.get(m_rows * c + r);
    }

  private:
    int m_rows, m_cols;
    Array<float> data;
};

} // namespace data
