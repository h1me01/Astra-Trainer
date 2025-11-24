#include "select.h"

namespace kernel {

constexpr int block_size = 1024;

template <bool UseActivation>
__global__ void select_fwd_kernel( //
    const float *in_v,             //
    float *linear_out,             //
    float *activated,              //
    const int *indices,            //
    const int in_r,                //
    const int out_r,               //
    const int batch_size,          //
    const ActivationType act_type  //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch_size * out_r)
        return;

    const int batch_idx = idx / out_r;
    const int out_idx = idx % out_r;

    const int bucket = indices[batch_idx];
    const int in_offset = in_r * batch_idx + out_r * bucket + out_idx;

    const float in_value = in_v[in_offset];

    const int out_offset = out_r * batch_idx + out_idx;

    linear_out[out_offset] = in_value;
    if(UseActivation)
        activated[out_offset] = activate_fwd(in_value, act_type);
}

void select_fwd(                  //
    const DenseMatrix &in_v,      //
    DenseMatrix &linear_out,      //
    DenseMatrix &activated,       //
    const Array<int> &indices,    //
    const ActivationType act_type //
) {
    ASSERT(in_v.cols() == linear_out.cols());
    ASSERT(linear_out.cols() == indices.size());

    ASSERT(in_v.is_dev_allocated() &&       //
           linear_out.is_dev_allocated() && //
           indices.is_dev_allocated());

    const int blocks = get_num_blocks(linear_out.size(), block_size);
    if(has_activation(act_type)) {
        ASSERT(activated.is_dev_allocated());

        select_fwd_kernel<true><<<blocks, block_size>>>( //
            in_v.dev_address(),
            linear_out.dev_address(),
            activated.dev_address(),
            indices.dev_address(),
            in_v.rows(),
            linear_out.rows(),
            linear_out.cols(),
            act_type);
    } else {
        select_fwd_kernel<false><<<blocks, block_size>>>( //
            in_v.dev_address(),
            linear_out.dev_address(),
            activated.dev_address(),
            indices.dev_address(),
            in_v.rows(),
            linear_out.rows(),
            linear_out.cols(),
            act_type);
    }
}

} // namespace kernel
