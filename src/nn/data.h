#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <new>
#include <ostream>
#include <random>
#include <sstream>
#include <string>

#include "../misc.h"

// Array

template <typename T> //
class Array {
  private:
    int m_size = 0;
    T *host_data = nullptr;
    T *dev_data = nullptr;

  public:
    Array() : m_size(0) {
        host_data = nullptr;
        dev_data = nullptr;
    }

    explicit Array(int size) : m_size(size) {
        alloc_host();
        alloc_dev();
    }

    Array(const Array<T> &other) : m_size(other.m_size) {
        if(other.is_host_allocated()) {
            alloc_host();
            copy_from_host(other.host_data, other.m_size);
        }

        if(other.is_dev_allocated()) {
            alloc_dev();
            copy_from_dev(other.dev_data, other.m_size);
        }
    }

    Array<T> &operator=(const Array<T> &other) {
        if(this != &other) {
            free_host();
            free_dev();
            m_size = other.m_size;

            if(other.is_host_allocated()) {
                alloc_host();
                copy_from_host(other.host_data, other.m_size);
            }

            if(other.is_dev_allocated()) {
                alloc_dev();
                copy_from_dev(other.dev_data, other.m_size);
            }
        }

        return *this;
    }

    virtual ~Array() {
        free_host();
        free_dev();
    }

    void free_host() {
        if(!is_host_allocated())
            return;
        delete[] host_data;
        host_data = nullptr;
    }

    void free_dev() {
        if(!is_dev_allocated())
            return;
        CUDA_ASSERT(cudaFree(dev_data));
        dev_data = nullptr;
    }

    void alloc_host() {
        if(m_size <= 0)
            return;
        if(is_host_allocated())
            free_host();
        host_data = new T[m_size]();
    }

    void alloc_dev() {
        if(m_size <= 0)
            return;
        if(is_dev_allocated())
            free_dev();
        CUDA_ASSERT(cudaMalloc(&dev_data, m_size * sizeof(T)));
    }

    void copy_from_host(const T *data, int size) {
        ASSERT(size == m_size);
        if(host_data == nullptr)
            alloc_host();
        memcpy(host_data, data, sizeof(T) * size);
    }

    void copy_from_dev(const T *data, int size) {
        ASSERT(size == m_size);
        if(dev_data == nullptr)
            alloc_dev();
        CUDA_ASSERT(cudaMemcpy(dev_data, data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
    }

    bool is_host_allocated() const {
        return host_data != nullptr;
    }

    bool is_dev_allocated() const {
        return dev_data != nullptr;
    }

    T *host_address() const {
        return host_data;
    }

    T *dev_address() const {
        return dev_data;
    }

    void clear_host() {
        if(host_data != nullptr)
            memset(host_data, 0, sizeof(T) * m_size);
    }

    void clear_dev() {
        if(dev_data != nullptr)
            CUDA_ASSERT(cudaMemset(dev_data, 0, sizeof(T) * m_size));
    }

    void host_to_dev() {
        if(!is_host_allocated() || !is_dev_allocated())
            return;
        CUDA_ASSERT(cudaMemcpy(dev_data, host_data, m_size * sizeof(T), cudaMemcpyHostToDevice));
    }

    void dev_to_host() {
        if(!is_host_allocated() || !is_dev_allocated())
            return;
        CUDA_ASSERT(cudaMemcpy(host_data, dev_data, m_size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T get(int idx) const {
        ASSERT(is_host_allocated());
        ASSERT(idx >= 0 && idx < m_size);
        return host_data[idx];
    }

    T &get(int idx) {
        ASSERT(is_host_allocated());
        ASSERT(idx >= 0 && idx < m_size);
        return host_data[idx];
    }

    T operator()(int idx) const {
        return get(idx);
    }

    T &operator()(int idx) {
        return get(idx);
    }

    int size() const {
        return m_size;
    }
};

// Dense Matrix

class DenseMatrix : public Array<float> {
  private:
    int n_rows, n_cols;

  public:
    using Array<float>::operator();

    DenseMatrix(int num_rows, int num_cols) //
        : n_rows(num_rows), n_cols(num_cols), Array(num_rows * num_cols) {}

    DenseMatrix(const DenseMatrix &other) //
        : n_rows(other.n_rows), n_cols(other.n_cols), Array<float>(other) {}

    DenseMatrix &operator=(const DenseMatrix &other) {
        if(this != &other) {
            n_rows = other.n_rows;
            n_cols = other.n_cols;
            Array<float>::operator=(other);
        }

        return *this;
    }

    int num_rows() const {
        return n_rows;
    }

    int num_cols() const {
        return n_cols;
    }

    float operator()(int row_idx, int col_idx) const {
        return get(num_rows() * col_idx + row_idx);
    }

    float &operator()(int row_idx, int col_idx) {
        return get(num_rows() * col_idx + row_idx);
    }
};

// Sparse Matrix

class SparseBatch {
  private:
    int batch_size;
    int max_entries;

    Array<int> psqt_indices;

    Array<int> feature_sizes;
    std::vector<Array<int>> features;

  public:
    SparseBatch(int batch_size, int max_entries) : batch_size(batch_size), max_entries(max_entries) {
        psqt_indices = Array<int>(batch_size);
        feature_sizes = Array<int>(batch_size);
        features.emplace_back(batch_size * max_entries); // stm_features
        features.emplace_back(batch_size * max_entries); // nstm_features
    }

    SparseBatch(const SparseBatch &other)
        : batch_size(other.batch_size),       //
          max_entries(other.max_entries),     //
          psqt_indices(other.psqt_indices),   //
          feature_sizes(other.feature_sizes), //
          features(other.features) {}

    SparseBatch &operator=(const SparseBatch &other) {
        if(this != &other) {
            batch_size = other.batch_size;
            max_entries = other.max_entries;
            psqt_indices = other.psqt_indices;
            feature_sizes = other.feature_sizes;
            features = other.features;
        }
        return *this;
    }

    int get_batch_size() const {
        return batch_size;
    }

    int get_max_entries() const {
        return max_entries;
    }

    Array<int> &get_psqt_indices() {
        return psqt_indices;
    }

    Array<int> &get_feature_sizes() {
        return feature_sizes;
    }

    std::vector<Array<int>> &get_features() {
        return features;
    }

    void host_to_dev() {
        psqt_indices.host_to_dev();
        feature_sizes.host_to_dev();
        for(auto &feature : features)
            feature.host_to_dev();
    }
};

// Tensor

class Tensor {
  private:
    DenseMatrix values;
    DenseMatrix gradients;

    float min_val = std::numeric_limits<float>::lowest();
    float max_val = std::numeric_limits<float>::max();

  public:
    Tensor(int num_rows, int num_cols)
        : values(DenseMatrix{num_rows, num_cols}), gradients(DenseMatrix{num_rows, num_cols}) {
        values.clear_host();
        gradients.clear_host();
    }

    Tensor(const Tensor &other) //
        : values(other.values), gradients(other.gradients) {}

    Tensor &operator=(const Tensor &other) {
        if(this != &other) {
            values = other.values;
            gradients = other.gradients;
        }
        return *this;
    }

    void init_uniformly() {
        std::mt19937 gen{std::random_device{}()};
        std::uniform_real_distribution<> dis(-0.1f, 0.1f);
        for(int i = 0; i < values.size(); i++)
            values(i) = dis(gen);
        values.host_to_dev();
    }

    void he_init(int previous_size) {
        std::mt19937 gen{std::random_device{}()};
        std::uniform_real_distribution<> dis(0, std::sqrt(2.0f / previous_size));
        for(int i = 0; i < values.size(); i++)
            values(i) = dis(gen);
        values.host_to_dev();
    }

    template <typename T> //
    void quant(FILE *f, float scale, bool trans = false) {
        static_assert(std::is_integral_v<T>, "quantize only supports integral types");

        auto quantize_value = [&](float orig) {
            T quant = static_cast<T>(round(orig * scale));
            if(quant < std::numeric_limits<T>::min() || quant > std::numeric_limits<T>::max()) {
                std::cout << "Overflow/Underflow detected while quantizing: quant = " << quant << " | orig = " << orig
                          << "\n";
                exit(1);
            }
            return quant;
        };

        Array<T> data{values.size()};
        int idx = 0;

        if(trans) {
            for(int r = 0; r < values.num_rows(); r++)
                for(int c = 0; c < values.num_cols(); c++)
                    data(idx++) = quantize_value(values(r, c));
        } else {
            for(int c = 0; c < values.num_cols(); c++)
                for(int r = 0; r < values.num_rows(); r++)
                    data(idx++) = quantize_value(values(r, c));
        }

        fwrite(data.host_address(), sizeof(T), data.size(), f);
    }

    void clamp(float min_val, float max_val) {
        this->min_val = min_val;
        this->max_val = max_val;
    }

    float min() const {
        return min_val;
    }

    float max() const {
        return max_val;
    }

    DenseMatrix &get_vals() {
        return values;
    }

    DenseMatrix &get_grads() {
        return gradients;
    }
};
