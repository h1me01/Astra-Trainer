#include "select.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void select_fwd_kernel( //
    const float *in_v,             //
    float *out_v,                  //
    const int *indices,            //
    const int in_r,                //
    const int out_r,               //
    const int batch_size           //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch_size * out_r)
        return;

    const int batch_idx = idx / out_r;
    const int out_idx = idx % out_r;

    const int bucket = indices[batch_idx];
    const int in_offset = in_r * batch_idx + out_r * bucket + out_idx;

    out_v[out_r * batch_idx + out_idx] = in_v[in_offset];
}

void select_fwd(              //
    const DenseMatrix &in_v,  //
    DenseMatrix &out_v,       //
    const Array<int> &indices //
) {
    ASSERT(in_v.cols() == out_v.cols());
    ASSERT(out_v.cols() == indices.size());

    ASSERT(in_v.is_dev_allocated() &&  //
           out_v.is_dev_allocated() && //
           indices.is_dev_allocated());

    const int blocks = get_num_blocks(out_v.size(), block_size);
    select_fwd_kernel<<<blocks, block_size>>>( //
        in_v.dev_address(),
        out_v.dev_address(),
        indices.dev_address(),
        in_v.rows(),
        out_v.rows(),
        out_v.cols());
}

} // namespace kernel
