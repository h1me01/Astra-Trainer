#include "select.h"

namespace kernel {

constexpr int block_size = 1024;

template <bool UseActivation>
__global__ void select_bwd_kernel( //
    float *in_g,                   //
    const float *linear_out,       //
    const float *grads,            //
    const int *indices,            //
    const int in_r,                //
    const int out_r,               //
    const int batch_size,          //
    const Activation act_type) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch_size * out_r)
        return;

    const int batch_idx = idx / out_r;
    const int out_idx = idx % out_r;

    const int bucket = indices[batch_idx];
    const int in_offset = in_r * batch_idx + out_r * bucket + out_idx;

    const int out_offset = out_r * batch_idx + out_idx;

    float grad = grads[out_offset];
    if(UseActivation)
        grad *= activate_bwd(linear_out[out_offset], act_type);

    in_g[in_offset] += grad;
}

void select_bwd(                   //
    DenseMatrix &in_g,             //
    const DenseMatrix &linear_out, //
    const DenseMatrix &grads,      //
    const Array<int> &indices,     //
    const Activation act_type  //
) {
    ASSERT(in_g.cols() == grads.cols() && grads.cols() == indices.size());

    ASSERT(in_g.is_dev_allocated() &&    //
           grads.is_dev_allocated() &&   //
           linear_out.is_dev_allocated() //
           && indices.is_dev_allocated());

    const int blocks = get_num_blocks(grads.size(), block_size);
    if(has_activation(act_type)) {
        select_bwd_kernel<true><<<blocks, block_size>>>( //
            in_g.dev_address(),
            linear_out.dev_address(),
            grads.dev_address(),
            indices.dev_address(),
            in_g.rows(),
            linear_out.rows(),
            linear_out.cols(),
            act_type);
    } else {
        select_bwd_kernel<false><<<blocks, block_size>>>( //
            in_g.dev_address(),
            linear_out.dev_address(),
            grads.dev_address(),
            indices.dev_address(),
            in_g.rows(),
            linear_out.rows(),
            linear_out.cols(),
            act_type);
    }
}

} // namespace kernel
