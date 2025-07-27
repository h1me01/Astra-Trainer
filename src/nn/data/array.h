#pragma once

#include "../../misc.h"

template <typename T> //
class Array {
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
            from_host(other.host_data, other.m_size);
        }

        if(other.is_dev_allocated()) {
            alloc_dev();
            from_dev(other.dev_data, other.m_size);
        }
    }

    Array<T> &operator=(const Array<T> &other) {
        if(this != &other) {
            free_host();
            free_dev();
            m_size = other.m_size;

            if(other.is_host_allocated()) {
                alloc_host();
                from_host(other.host_data, other.m_size);
            }

            if(other.is_dev_allocated()) {
                alloc_dev();
                from_dev(other.dev_data, other.m_size);
            }
        }

        return *this;
    }

    virtual ~Array() {
        free_host();
        free_dev();
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

    void clear() {
        clear_host();
        clear_dev();
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

  private:
    int m_size = 0;
    T *host_data = nullptr;
    T *dev_data = nullptr;

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

    void from_host(const T *data, int size) {
        ASSERT(size == m_size);
        if(host_data == nullptr)
            alloc_host();
        memcpy(host_data, data, sizeof(T) * size);
    }

    void from_dev(const T *data, int size) {
        ASSERT(size == m_size);
        if(dev_data == nullptr)
            alloc_dev();
        CUDA_ASSERT(cudaMemcpy(dev_data, data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
    }
};