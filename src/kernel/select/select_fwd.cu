#include "select.h"

namespace kernel {

constexpr int num_threads = 256;

template <Activation act_type>
__global__ void select_fwd_kernel(
    const float* in_d, float* out_d, const int* indices, const int in_r, const int out_r, const int batch_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_r)
        return;

    const int batch_idx = idx / out_r;
    const int out_idx = idx % out_r;

    const int bucket = indices[batch_idx];
    const int in_offset = in_r * batch_idx + out_r * bucket + out_idx;

    const float in_value = in_d[in_offset];

    const int out_offset = out_r * batch_idx + out_idx;
    out_d[out_offset] = activate_fwd<act_type>(in_value);
}

void select_fwd(const DenseMatrix& in_d, DenseMatrix& out_d, const Array<int>& indices, const Activation act_type) {
    CHECK(in_d.cols() == out_d.cols());
    CHECK(out_d.cols() == indices.size());

    CHECK(
        in_d.is_dev_allocated() &&  //
        out_d.is_dev_allocated() && //
        indices.is_dev_allocated()
    );

    const int blocks = cuda::ceil_div(out_d.size(), num_threads);
    DISPATCH_ACTIVATION(
        act_type,
        select_fwd_kernel,
        <<<blocks, num_threads>>>(
            in_d.dev_address(), out_d.dev_address(), indices.dev_address(), in_d.rows(), out_d.rows(), out_d.cols()
        )
    );
}

} // namespace kernel
