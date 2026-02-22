#include "concat.h"
#include <cstdio>

namespace kernel {

constexpr int num_threads = 256;

template <Activation act_type>
__global__ void
concat_fwd_kernel(const float* in_d, float* out_d, const int out_r, const int in_r, const int batch_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_r * batch_size)
        return;

    const int batch_idx = idx / in_r;
    const int curr_in_r = idx % in_r;

    const int in_idx = curr_in_r + batch_idx * in_r;
    const int out_idx = curr_in_r + batch_idx * out_r;

    out_d[out_idx] = activate_fwd<act_type>(in_d[in_idx]);
}

void concat_fwd(const DenseMatrix& in_d, DenseMatrix& out_d, const int offset, const Activation act_type) {
    CHECK(in_d.cols() == out_d.cols());
    CHECK(in_d.is_dev_allocated() && out_d.is_dev_allocated());

    const int blocks = cuda::ceil_div(in_d.size(), num_threads);
    DISPATCH_ACTIVATION(
        act_type,
        concat_fwd_kernel,
        <<<blocks, num_threads>>>(
            in_d.dev_address(), out_d.dev_address() + offset, out_d.rows(), in_d.rows(), out_d.cols()
        )
    );
}

} // namespace kernel
