#pragma once

namespace kernel {

__device__ __forceinline__ float clamp(float x, float min, float max) {
    return fmaxf(min, fminf(x, max));
}

inline int get_num_blocks(int n_elements, int block_size) {
    return (n_elements + block_size - 1) / block_size;
}

template <typename T, typename U>
__device__ __forceinline__ const T* as_vec(const U* ptr) {
    return reinterpret_cast<const T*>(ptr);
}

template <typename T, typename U>
__device__ __forceinline__ T* as_vec(U* ptr) {
    return reinterpret_cast<T*>(ptr);
}

template <typename T>
__device__ __forceinline__ void add_t4(T& a, const T& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

} // namespace kernel
