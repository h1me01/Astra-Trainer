#pragma once

namespace kernel {

inline __device__ float clamp(float x, float min, float max) {
    return fmaxf(min, fminf(x, max));
}

inline int get_num_blocks(int n_elements, int block_size) {
    return (n_elements + block_size - 1) / block_size;
}

} // namespace kernel
