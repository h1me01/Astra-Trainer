#include "factorizer.h"

namespace kernel {

constexpr int num_threads = 1024;

__global__ void factorizer_bwd_kernel(float* in_g, const float* out_g, const int in_size, const int total_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;
    if (vec_idx >= total_size)
        return;

    const int f_idx = idx % (in_size / 4);

    if (vec_idx + 4 <= total_size) {
        const float4 out_val = as_vec<const float4>(out_g)[idx];

        float4* dst = &as_vec<float4>(in_g)[f_idx];
        atomicAdd(&dst->x, out_val.x);
        atomicAdd(&dst->y, out_val.y);
        atomicAdd(&dst->z, out_val.z);
        atomicAdd(&dst->w, out_val.w);
    } else {
        for (int i = vec_idx; i < total_size; i++)
            atomicAdd(&in_g[i % in_size], out_g[i]);
    }
}

void factorizer_bwd(DenseMatrix& in_g, const DenseMatrix& out_g) {
    CHECK(out_g.size() % in_g.size() == 0);
    CHECK(out_g.dev_address() && in_g.dev_address());

    const int blocks = cuda::ceil_div(out_g.size(), num_threads * 4);
    factorizer_bwd_kernel<<<blocks, num_threads>>>(in_g.dev_address(), out_g.dev_address(), in_g.size(), out_g.size());

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kernel
