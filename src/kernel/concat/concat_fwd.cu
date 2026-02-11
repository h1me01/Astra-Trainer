#include "concat.h"
#include <cstdio>

namespace kernel {

constexpr int block_size = 256;

template <Activation act_type>
__global__ void concat_fwd_kernel(
    const float* in1_v, const float* in2_v, float* out_d, const int out_r, const int in1_r, const int batch_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_r * batch_size)
        return;

    const int batch_idx = idx / out_r;
    const int out_idx = idx % out_r;

    const int out_offset = out_idx + batch_idx * out_r;

    float val;
    if (out_idx < in1_r) {
        const int in1_offset = out_idx + batch_idx * in1_r;
        val = in1_v[in1_offset];
    } else {
        const int in2_offset = (out_idx - in1_r) + batch_idx * (out_r - in1_r);
        val = in2_v[in2_offset];
    }

    out_d[out_offset] = activate_fwd<act_type>(val);
}

void concat_fwd(const DenseMatrix& in1_v, const DenseMatrix& in2_v, DenseMatrix& out_d, const Activation act_type) {
    ASSERT(
        in1_v.cols() == out_d.cols() && //
        in2_v.cols() == out_d.cols() && //
        in1_v.rows() + in2_v.rows() == out_d.rows()
    );

    ASSERT(
        in1_v.is_dev_allocated() && //
        in2_v.is_dev_allocated() && //
        out_d.is_dev_allocated()
    );

    const int blocks = get_num_blocks(out_d.size(), block_size);
    DISPATCH_ACTIVATION(
        act_type,
        concat_fwd_kernel,
        <<<blocks, block_size>>>(
            in1_v.dev_address(), in2_v.dev_address(), out_d.dev_address(), out_d.rows(), in1_v.rows(), out_d.cols()
        )
    );
}

} // namespace kernel
