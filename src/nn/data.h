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
    Array() : m_size(0) {}

    explicit Array(int size) : m_size(size) {
        allocHost();
        allocDev();
    }

    Array(const Array<T> &other) : m_size(other.m_size) {
        if(other.isHostAllocated()) {
            allocHost();
            copyFromHost(other.host_data, other.m_size);
        }

        if(other.isDevAllocated()) {
            allocDev();
            copyFromDev(other.dev_data, other.m_size);
        }
    }

    Array<T> &operator=(const Array<T> &other) {
        if(this != &other) {
            freeHost();
            freeDev();
            m_size = other.m_size;

            if(other.isHostAllocated()) {
                allocHost();
                copyFromHost(other.host_data, other.m_size);
            }

            if(other.isDevAllocated()) {
                allocDev();
                copyFromDev(other.dev_data, other.m_size);
            }
        }

        return *this;
    }

    virtual ~Array() {
        freeHost();
        freeDev();
    }

    void freeHost() {
        if(!isHostAllocated())
            return;
        delete[] host_data;
        host_data = nullptr;
    }

    void freeDev() {
        if(!isDevAllocated())
            return;
        CUDA_ASSERT(cudaFree(dev_data));
        dev_data = nullptr;
    }

    void allocHost() {
        if(m_size <= 0)
            return;
        if(isHostAllocated())
            freeHost();
        host_data = new T[m_size]();
    }

    void allocDev() {
        if(m_size <= 0)
            return;
        if(isDevAllocated())
            freeDev();
        CUDA_ASSERT(cudaMalloc(&dev_data, m_size * sizeof(T)));
    }

    void copyFromHost(const T *data, int size) {
        ASSERT(size == m_size);
        if(host_data == nullptr)
            allocHost();
        memcpy(host_data, data, sizeof(T) * size);
    }

    void copyFromDev(const T *data, int size) {
        ASSERT(size == m_size);
        if(dev_data == nullptr)
            allocDev();
        CUDA_ASSERT(cudaMemcpy(dev_data, data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
    }

    bool isHostAllocated() const {
        return host_data != nullptr;
    }
    bool isDevAllocated() const {
        return dev_data != nullptr;
    }

    T *hostAddress() const {
        return host_data;
    }
    T *devAddress() const {
        return dev_data;
    }

    void clearHost() {
        if(host_data != nullptr)
            memset(host_data, 0, sizeof(T) * m_size);
    }

    void clearDev() {
        if(dev_data != nullptr)
            CUDA_ASSERT(cudaMemset(dev_data, 0, sizeof(T) * m_size));
    }

    void hostToDev() {
        if(!isHostAllocated() || !isDevAllocated())
            return;
        CUDA_ASSERT(cudaMemcpy(dev_data, host_data, m_size * sizeof(T), cudaMemcpyHostToDevice));
    }

    void devToHost() {
        if(!isHostAllocated() || !isDevAllocated())
            return;
        CUDA_ASSERT(cudaMemcpy(host_data, dev_data, m_size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T get(int idx) const {
        ASSERT(isHostAllocated());
        ASSERT(idx >= 0 && idx < m_size);
        return host_data[idx];
    }

    T &get(int idx) {
        ASSERT(isHostAllocated());
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
    int num_rows, num_cols;

  public:
    using Array<float>::operator();

    DenseMatrix(int num_rows, int num_cols) //
        : num_rows(num_rows), num_cols(num_cols), Array(num_rows * num_cols) {}

    DenseMatrix(const DenseMatrix &other) //
        : num_rows(other.num_rows), num_cols(other.num_cols), Array<float>(other) {}

    DenseMatrix &operator=(const DenseMatrix &other) {
        if(this != &other) {
            num_rows = other.num_rows;
            num_cols = other.num_cols;
            Array<float>::operator=(other);
        }

        return *this;
    }

    int numRows() const {
        return num_rows;
    }

    int numCols() const {
        return num_cols;
    }

    float operator()(int row_idx, int col_idx) const {
        return get(numRows() * col_idx + row_idx);
    }

    float &operator()(int row_idx, int col_idx) {
        return get(numRows() * col_idx + row_idx);
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

    int getBatchSize() const {
        return batch_size;
    }

    int maxEntries() const {
        return max_entries;
    }

    Array<int> &getPSQTIndices() {
        return psqt_indices;
    }

    Array<int> &getFeatureSizes() {
        return feature_sizes;
    }

    std::vector<Array<int>> &getFeatures() {
        return features;
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
        values.clearHost();
        gradients.clearHost();
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

    void initUniformly() {
        std::mt19937 gen{std::random_device{}()};
        std::uniform_real_distribution<> dis(-0.1f, 0.1f);
        for(int i = 0; i < values.size(); i++)
            values(i) = dis(gen);
        values.hostToDev();
    }

    void heInit(int previous_size) {
        std::mt19937 gen{std::random_device{}()};
        std::uniform_real_distribution<> dis(0, std::sqrt(2.0f / previous_size));
        for(int i = 0; i < values.size(); i++)
            values(i) = dis(gen);
        values.hostToDev();
    }

    template <typename T> void quantize(FILE *f, float scale, bool transpose = false) {
        if(!std::is_integral_v<T>) {
            std::cout << "Error: quantize only supports integral types\n";
            exit(1);
        }

        Array<T> data{values.size()};
        data.allocHost();

        int num_rows = values.numRows();
        int num_cols = values.numCols();

        int idx = 0;
        for(int i = 0; i < (transpose ? num_cols : num_rows); i++)
            for(int j = 0; j < (transpose ? num_rows : num_cols); j++) {
                float orig = transpose ? values(j, i) : values(i, j);
                T quant = static_cast<T>(round(orig * scale));

                // check for overflow/underflow
                if(quant < std::numeric_limits<T>::min() || quant > std::numeric_limits<T>::max()) {
                    std::cout << "Overflow/Underflow detected while quantitizing: quant = " << quant
                              << " | orig = " << orig << "\n";
                    exit(1);
                }

                data(idx++) = quant;
            }

        fwrite(data.hostAddress(), sizeof(T), data.size(), f);
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

    DenseMatrix &getValues() {
        return values;
    }

    DenseMatrix &getGradients() {
        return gradients;
    }
};
