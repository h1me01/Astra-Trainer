#include "factorizer.h"

namespace kernel {

constexpr int num_threads = 1024;

__global__ void factorizer_fwd_kernel(
    const float* factorizer_d, const float* weights_d, float* out_d, const int factorizer_size, const int total_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;
    if (vec_idx >= total_size)
        return;

    // Map output position back to factorizer position (it tiles/repeats)
    const int f_idx = idx % (factorizer_size / 4);

    if (vec_idx + 4 <= total_size) {
        float4 f_val = as_vec<const float4>(factorizer_d)[f_idx];
        float4 w_val = as_vec<const float4>(weights_d)[idx];
        as_vec<float4>(out_d)[idx] =
            make_float4(f_val.x + w_val.x, f_val.y + w_val.y, f_val.z + w_val.z, f_val.w + w_val.w);
    } else {
        for (int i = vec_idx; i < total_size; i++)
            out_d[i] = factorizer_d[i % factorizer_size] + weights_d[i];
    }
}

void factorizer_fwd(const DenseMatrix& factorizer_d, const DenseMatrix& weights_d, DenseMatrix& out_d) {
    CHECK(weights_d.size() == out_d.size());
    CHECK(out_d.size() % factorizer_d.size() == 0);
    CHECK(factorizer_d.size() % 4 == 0);
    CHECK(out_d.dev_address() && factorizer_d.dev_address() && weights_d.dev_address());

    const int blocks = cuda::ceil_div(out_d.size(), num_threads * 4);
    factorizer_fwd_kernel<<<blocks, num_threads>>>(
        factorizer_d.dev_address(), weights_d.dev_address(), out_d.dev_address(), factorizer_d.size(), out_d.size()
    );
}

} // namespace kernel
