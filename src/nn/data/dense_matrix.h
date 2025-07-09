#pragma once

#include "array.h"

// column-major dense matrix
template <typename T> //
class DenseMatrix : public Array<T> {
  private:
    int m_rows, m_cols;

  public:
    using Array<T>::get;
    using Array<T>::operator();

    DenseMatrix(int rows, int cols) //
        : m_rows(rows), m_cols(cols), Array<T>(rows * cols) {}

    DenseMatrix(const DenseMatrix &other) //
        : m_rows(other.m_rows), m_cols(other.m_cols), Array<T>(other) {}

    DenseMatrix &operator=(const DenseMatrix &other) {
        if(this != &other) {
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            Array<T>::operator=(other);
        }

        return *this;
    }

    int rows() const {
        return m_rows;
    }

    int cols() const {
        return m_cols;
    }

    T operator()(int r, int c) const {
        return get(m_rows * c + r);
    }

    T &operator()(int r, int c) {
        return get(m_rows * c + r);
    }
};
