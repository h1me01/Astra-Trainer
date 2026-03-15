#include "pairwise_mul.h"

namespace kernel {

constexpr int num_threads = 256;

__global__ void pairwise_mul_fwd_kernel(
    const float* in_d, float* out_d, const int feature_size, const int out_r, const int batch_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= feature_size * batch_size)
        return;

    const int batch_idx = idx / feature_size;
    const int feature_idx = idx % feature_size;

    const int in_offset_a = batch_idx * 2 * feature_size + feature_idx;
    const int in_offset_b = in_offset_a + feature_size;

    const int out_idx = batch_idx * out_r + feature_idx;

    out_d[out_idx] = in_d[in_offset_a] * in_d[in_offset_b];
}

void pairwise_mul_fwd(const DenseMatrix& in_d, DenseMatrix& out_d) {
    const int feature_size = in_d.rows() / 2;

    CHECK(
        in_d.rows() % 2 == 0 &&        //
        in_d.cols() == out_d.cols() && //
        out_d.rows() == feature_size
    );

    CHECK(in_d.is_dev_allocated() && out_d.is_dev_allocated());

    const int blocks = cuda::ceil_div(feature_size * in_d.cols(), num_threads);
    pairwise_mul_fwd_kernel<<<blocks, num_threads>>>(
        in_d.dev_address(), out_d.dev_address(), feature_size, out_d.rows(), in_d.cols()
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kernel
