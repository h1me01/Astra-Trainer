#include "select.h"

namespace kernel {

constexpr int block_size = 256;

template <Activation act_type>
__global__ void select_fwd_kernel(
    const float* in_v, float* out_d, const int* indices, const int in_r, const int out_r, const int batch_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_r)
        return;

    const int batch_idx = idx / out_r;
    const int out_idx = idx % out_r;

    const int bucket = indices[batch_idx];
    const int in_offset = in_r * batch_idx + out_r * bucket + out_idx;

    const float in_value = in_v[in_offset];

    const int out_offset = out_r * batch_idx + out_idx;
    out_d[out_offset] = activate_fwd<act_type>(in_value);
}

void select_fwd(const DenseMatrix& in_v, DenseMatrix& out_d, const Array<int>& indices, const Activation act_type) {
    ASSERT(in_v.cols() == out_d.cols());
    ASSERT(out_d.cols() == indices.size());

    ASSERT(
        in_v.is_dev_allocated() &&  //
        out_d.is_dev_allocated() && //
        indices.is_dev_allocated()
    );

    const int blocks = get_num_blocks(out_d.size(), block_size);
    DISPATCH_ACTIVATION(
        act_type,
        select_fwd_kernel,
        <<<blocks, block_size>>>(
            in_v.dev_address(), out_d.dev_address(), indices.dev_address(), in_v.rows(), out_d.rows(), out_d.cols()
        )
    );
}

} // namespace kernel
