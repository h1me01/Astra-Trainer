#pragma once

#include "../misc.h"

namespace data {

template <typename T>
class CudaDevicePtr {
  public:
    CudaDevicePtr() = default;

    explicit CudaDevicePtr(size_t count) {
        if (count > 0)
            CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    }

    ~CudaDevicePtr() {
        if (ptr)
            cudaFree(ptr);
    }

    CudaDevicePtr(const CudaDevicePtr&) = delete;
    CudaDevicePtr& operator=(const CudaDevicePtr&) = delete;

    CudaDevicePtr(CudaDevicePtr&& other) noexcept
        : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    CudaDevicePtr& operator=(CudaDevicePtr&& other) noexcept {
        if (this != &other) {
            if (ptr)
                cudaFree(ptr);

            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr; }

    explicit operator bool() const { return ptr != nullptr; }

  private:
    T* ptr = nullptr;
};

template <typename T>
class CudaHostPtr {
  public:
    CudaHostPtr() = default;

    explicit CudaHostPtr(size_t count) {
        if (count > 0)
            CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
    }

    ~CudaHostPtr() {
        if (ptr)
            cudaFreeHost(ptr);
    }

    CudaHostPtr(const CudaHostPtr&) = delete;
    CudaHostPtr& operator=(const CudaHostPtr&) = delete;

    CudaHostPtr(CudaHostPtr&& other) noexcept
        : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    CudaHostPtr& operator=(CudaHostPtr&& other) noexcept {
        if (this != &other) {
            if (ptr)
                cudaFreeHost(ptr);

            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr; }

    explicit operator bool() const { return ptr != nullptr; }

  private:
    T* ptr = nullptr;
};

template <typename T>
class Array {
  public:
    Array() = default;

    explicit Array(int size, bool use_pinned = false)
        : m_size(size),
          use_pinned(use_pinned) {

        if (size > 0) {
            if (use_pinned)
                pinned_host_data = CudaHostPtr<T>(size);
            else
                host_data = std::make_unique<T[]>(size);

            dev_data = CudaDevicePtr<T>(size);
        }
    }

    Array(const Array<T>& other)
        : m_size(other.m_size),
          use_pinned(other.use_pinned) {

        if (use_pinned && other.pinned_host_data) {
            pinned_host_data = CudaHostPtr<T>(m_size);
            std::memcpy(pinned_host_data.get(), other.pinned_host_data.get(), m_size * sizeof(T));
        } else if (other.host_data) {
            host_data = std::make_unique<T[]>(m_size);
            std::memcpy(host_data.get(), other.host_data.get(), m_size * sizeof(T));
        }

        if (other.dev_data) {
            dev_data = CudaDevicePtr<T>(m_size);
            CUDA_CHECK(cudaMemcpy(dev_data.get(), other.dev_data.get(), m_size * sizeof(T), cudaMemcpyDeviceToDevice));
        }
    }

    Array(Array<T>&& other) noexcept = default;

    Array<T>& operator=(const Array<T>& other) {
        if (this != &other) {
            Array<T> temp(other);
            *this = std::move(temp);
        }
        return *this;
    }

    Array<T>& operator=(Array<T>&& other) noexcept = default;

    virtual ~Array() = default;

    bool is_host_allocated() const { return host_data != nullptr || pinned_host_data; }
    bool is_dev_allocated() const { return static_cast<bool>(dev_data); }

    T* host_address() const { return use_pinned ? pinned_host_data.get() : host_data.get(); }
    T* dev_address() const { return dev_data.get(); }

    void free_dev() { dev_data = CudaDevicePtr<T>(); }

    void free_host() {
        host_data.reset();
        pinned_host_data = CudaHostPtr<T>();
    }

    void free() {
        free_host();
        free_dev();
    }

    void clear() {
        clear_host();
        clear_dev();
    }

    void clear_host() {
        if (use_pinned && pinned_host_data)
            std::memset(pinned_host_data.get(), 0, sizeof(T) * m_size);
        else if (host_data)
            std::memset(host_data.get(), 0, sizeof(T) * m_size);
    }

    void clear_dev() {
        if (dev_data)
            CUDA_CHECK(cudaMemsetAsync(dev_data.get(), 0, sizeof(T) * m_size, 0));
    }

    void host_to_dev() {
        if (!is_host_allocated() || !is_dev_allocated())
            return;
        T* h_ptr = use_pinned ? pinned_host_data.get() : host_data.get();
        CUDA_CHECK(cudaMemcpy(dev_data.get(), h_ptr, m_size * sizeof(T), cudaMemcpyHostToDevice));
    }

    void host_to_dev_async(cudaStream_t stream = 0) {
        if (!is_host_allocated() || !is_dev_allocated())
            return;
        T* h_ptr = use_pinned ? pinned_host_data.get() : host_data.get();
        if (use_pinned) {
            CUDA_CHECK(cudaMemcpyAsync(dev_data.get(), h_ptr, m_size * sizeof(T), cudaMemcpyHostToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpy(dev_data.get(), h_ptr, m_size * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void dev_to_host() {
        if (!is_host_allocated() || !is_dev_allocated())
            return;
        T* h_ptr = use_pinned ? pinned_host_data.get() : host_data.get();
        CUDA_CHECK(cudaMemcpy(h_ptr, dev_data.get(), m_size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T get(int idx) const {
        CHECK(is_host_allocated());
        CHECK(idx >= 0 && idx < m_size);
        T* h_ptr = use_pinned ? pinned_host_data.get() : host_data.get();
        return h_ptr[idx];
    }

    T& get(int idx) {
        CHECK(is_host_allocated());
        CHECK(idx >= 0 && idx < m_size);
        T* h_ptr = use_pinned ? pinned_host_data.get() : host_data.get();
        return h_ptr[idx];
    }

    T operator()(int idx) const { return get(idx); }
    T& operator()(int idx) { return get(idx); }

    int size() const { return m_size; }

  protected:
    int m_size = 0;
    bool use_pinned = false;
    Ptr<T[]> host_data;
    CudaHostPtr<T> pinned_host_data;
    CudaDevicePtr<T> dev_data;
};

} // namespace data
