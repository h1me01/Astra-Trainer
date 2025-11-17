#include "pairwise_mul.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void pairwise_mul_bwd_kernel( //
    const float *in_v,                   //
    float *in_g,                         //
    const float *out_g,                  //
    const int out_r,                     //
    const int batch_size                 //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= out_r * batch_size)
        return;

    const int batch_idx = idx / out_r;
    const int out_idx = idx % out_r;

    const int out_offset = batch_idx * out_r + out_idx;

    const int in_offset = batch_idx * 2 * out_r + out_idx;
    const int in_offset_plus = in_offset + out_r;

    const float grad = out_g[out_offset];

    in_g[in_offset] += grad * in_v[in_offset_plus];
    in_g[in_offset_plus] += grad * in_v[in_offset];
}

void pairwise_mul_bwd(Tensor &in, const DenseMatrix &out_g) {
    const auto &in_v = in.get_values();
    auto &in_g = in.get_gradients();

    ASSERT(in_v.cols() == out_g.cols() && in_v.rows() == 2 * out_g.rows());

    ASSERT(in_v.is_dev_allocated() && //
           in_g.is_dev_allocated() && //
           out_g.is_dev_allocated());

    const int blocks = get_num_blocks(out_g.size(), block_size);
    pairwise_mul_bwd_kernel<<<blocks, block_size>>>( //
        in_v.dev_address(),
        in_g.dev_address(),
        out_g.dev_address(),
        out_g.rows(),
        out_g.cols());
}

} // namespace kernel
