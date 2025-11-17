#include "pairwise_mul.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void pairwise_mul_fwd_kernel( //
    const float *in,                     //
    float *out,                          //
    const int out_r,                     //
    const int batch_size                 //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= out_r * batch_size)
        return;

    const int batch_idx = idx / out_r;
    const int out_idx = idx % out_r;

    const int in_offset = batch_idx * 2 * out_r + out_idx;
    const int out_offset = batch_idx * out_r + out_idx;

    out[out_offset] = in[in_offset] * in[in_offset + out_r];
}

void pairwise_mul_fwd(const DenseMatrix &in_v, DenseMatrix &out_v) {
    ASSERT(in_v.cols() == out_v.cols() && in_v.rows() == 2 * out_v.rows());

    ASSERT(in_v.is_dev_allocated() && out_v.is_dev_allocated());

    const int blocks = get_num_blocks(out_v.size(), block_size);
    pairwise_mul_fwd_kernel<<<blocks, block_size>>>( //
        in_v.dev_address(),
        out_v.dev_address(),
        out_v.rows(),
        out_v.cols());
}

} // namespace kernel
