#pragma once

#include <cuda/cmath>

namespace kernel {

__device__ __forceinline__ float clamp(float x, float min, float max) {
    return fmaxf(min, fminf(x, max));
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
