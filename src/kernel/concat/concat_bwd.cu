#include "concat.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void concat_bwd_kernel( //
    const float *out_g,            //
    float *in1_g,                  //
    float *in2_g,                  //
    const int in1_r,               //
    const int out_r,               //
    const int batch_size           //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= out_r * batch_size)
        return;

    const int batch_idx = idx / out_r;
    const int out_idx = idx % out_r;

    const int out_offset = out_idx + batch_idx * out_r;

    if(out_idx < in1_r) {
        const int in1_offset = out_idx + batch_idx * in1_r;
        in1_g[in1_offset] = out_g[out_offset];
    } else {
        const int in2_offset = (out_idx - in1_r) + batch_idx * (out_r - in1_r);
        in2_g[in2_offset] = out_g[out_offset];
    }
}

void concat_bwd(             //
    DenseMatrix &in1_g,      //
    DenseMatrix &in2_g,      //
    const DenseMatrix &out_g //
) {
    ASSERT(in1_g.cols() == out_g.cols() && //
           in2_g.cols() == out_g.cols() && //
           in1_g.rows() + in2_g.rows() == out_g.rows());

    ASSERT(in1_g.is_dev_allocated() && //
           in2_g.is_dev_allocated() && //
           out_g.is_dev_allocated());

    const int blocks = get_num_blocks(out_g.size(), block_size);
    concat_bwd_kernel<<<blocks, block_size>>>( //
        out_g.dev_address(),
        in1_g.dev_address(),
        in2_g.dev_address(),
        in1_g.rows(),
        out_g.rows(),
        out_g.cols());
}

} // namespace kernel
