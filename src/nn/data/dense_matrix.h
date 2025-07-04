#pragma once

#include "array.h"

template <typename T> //
class DenseMatrix : public Array<T> {
  private:
    int n_rows, n_cols;

  public:
    using Array<T>::operator();

    DenseMatrix(int num_rows, int num_cols) //
        : n_rows(num_rows), n_cols(num_cols), Array<T>(num_rows * num_cols) {}

    DenseMatrix(const DenseMatrix &other) //
        : n_rows(other.n_rows), n_cols(other.n_cols), Array<T>(other) {}

    DenseMatrix &operator=(const DenseMatrix &other) {
        if(this != &other) {
            n_rows = other.n_rows;
            n_cols = other.n_cols;
            Array<T>::operator=(other);
        }

        return *this;
    }

    int num_rows() const {
        return n_rows;
    }

    int num_cols() const {
        return n_cols;
    }

    T operator()(int row_idx, int col_idx) const {
        return get(num_rows() * col_idx + row_idx);
    }

    T &operator()(int row_idx, int col_idx) {
        return get(num_rows() * col_idx + row_idx);
    }
};
