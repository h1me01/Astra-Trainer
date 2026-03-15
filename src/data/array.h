#pragma once

#include "../misc.h"

namespace data {

template <typename T>
class CudaDevicePtr {
  public:
    CudaDevicePtr() = default;

    explicit CudaDevicePtr(size_t count) {
        if (count > 0)
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    ~CudaDevicePtr() {
        if (ptr_)
            cudaFree(ptr_);
    }

    CudaDevicePtr(const CudaDevicePtr&) = delete;
    CudaDevicePtr& operator=(const CudaDevicePtr&) = delete;

    CudaDevicePtr(CudaDevicePtr&& other) noexcept
        : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    CudaDevicePtr& operator=(CudaDevicePtr&& other) noexcept {
        if (this != &other) {
            if (ptr_)
                cudaFree(ptr_);

            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr_; }

    explicit operator bool() const { return ptr_ != nullptr; }

  private:
    T* ptr_ = nullptr;
};

template <typename T>
class CudaHostPtr {
  public:
    CudaHostPtr() = default;

    explicit CudaHostPtr(size_t count) {
        if (count > 0)
            CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
    }

    ~CudaHostPtr() {
        if (ptr_)
            cudaFreeHost(ptr_);
    }

    CudaHostPtr(const CudaHostPtr&) = delete;
    CudaHostPtr& operator=(const CudaHostPtr&) = delete;

    CudaHostPtr(CudaHostPtr&& other) noexcept
        : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    CudaHostPtr& operator=(CudaHostPtr&& other) noexcept {
        if (this != &other) {
            if (ptr_)
                cudaFreeHost(ptr_);

            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr_; }

    explicit operator bool() const { return ptr_ != nullptr; }

  private:
    T* ptr_ = nullptr;
};

template <typename T>
class Array {
  public:
    Array() = default;

    explicit Array(int size, bool pinned = false)
        : size_(size),
          pinned_(pinned) {

        if (size > 0) {
            if (pinned)
                pinned_host_data_ = CudaHostPtr<T>(size);
            else
                host_data_ = std::make_unique<T[]>(size);

            dev_data_ = CudaDevicePtr<T>(size);
        }
    }

    Array(const Array<T>& other)
        : size_(other.size_),
          pinned_(other.pinned_) {

        if (pinned_ && other.pinned_host_data_) {
            pinned_host_data_ = CudaHostPtr<T>(size_);
            std::memcpy(pinned_host_data_.get(), other.pinned_host_data_.get(), size_ * sizeof(T));
        } else if (other.host_data_) {
            host_data_ = std::make_unique<T[]>(size_);
            std::memcpy(host_data_.get(), other.host_data_.get(), size_ * sizeof(T));
        }

        if (other.dev_data_) {
            dev_data_ = CudaDevicePtr<T>(size_);
            CUDA_CHECK(cudaMemcpy(dev_data_.get(), other.dev_data_.get(), size_ * sizeof(T), cudaMemcpyDeviceToDevice));
        }
    }

    Array<T>& operator=(Array<T>&& other) noexcept = default;

    Array<T>& operator=(const Array<T>& other) {
        if (this != &other) {
            Array<T> temp(other);
            *this = std::move(temp);
        }
        return *this;
    }

    bool is_host_allocated() const { return host_data_ != nullptr || pinned_host_data_; }
    bool is_dev_allocated() const { return static_cast<bool>(dev_data_); }

    T* host_address() const { return pinned_ ? pinned_host_data_.get() : host_data_.get(); }
    T* dev_address() const { return dev_data_.get(); }

    void free_dev() { dev_data_ = CudaDevicePtr<T>(); }

    void free_host() {
        host_data_.reset();
        pinned_host_data_ = CudaHostPtr<T>();
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
        if (pinned_ && pinned_host_data_)
            std::memset(pinned_host_data_.get(), 0, sizeof(T) * size_);
        else if (host_data_)
            std::memset(host_data_.get(), 0, sizeof(T) * size_);
    }

    void clear_dev() {
        if (dev_data_)
            CUDA_CHECK(cudaMemsetAsync(dev_data_.get(), 0, sizeof(T) * size_, 0));
    }

    void host_to_dev() {
        if (!is_host_allocated() || !is_dev_allocated())
            return;
        T* h_ptr = pinned_ ? pinned_host_data_.get() : host_data_.get();
        CUDA_CHECK(cudaMemcpy(dev_data_.get(), h_ptr, size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    void dev_to_host() {
        if (!is_host_allocated() || !is_dev_allocated())
            return;
        T* h_ptr = pinned_ ? pinned_host_data_.get() : host_data_.get();
        CUDA_CHECK(cudaMemcpy(h_ptr, dev_data_.get(), size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T get(int idx) const {
        CHECK(is_host_allocated());
        CHECK(idx >= 0 && idx < size_);
        T* h_ptr = pinned_ ? pinned_host_data_.get() : host_data_.get();
        return h_ptr[idx];
    }

    T& get(int idx) {
        CHECK(is_host_allocated());
        CHECK(idx >= 0 && idx < size_);
        T* h_ptr = pinned_ ? pinned_host_data_.get() : host_data_.get();
        return h_ptr[idx];
    }

    T operator()(int idx) const { return get(idx); }
    T& operator()(int idx) { return get(idx); }

    int size() const { return size_; }

  protected:
    int size_ = 0;
    bool pinned_ = false;
    UPtr<T[]> host_data_;
    CudaHostPtr<T> pinned_host_data_;
    CudaDevicePtr<T> dev_data_;
};

} // namespace data
