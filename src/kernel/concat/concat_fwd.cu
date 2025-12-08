#include "concat.h"
#include <cstdio>

namespace kernel {

constexpr int block_size = 1024;

template <bool UseActivation>
__global__ void concat_fwd_kernel( //
    const float *in1_v,            //
    const float *in2_v,            //
    float *linear_out,             //
    float *activated,              //
    const int out_r,               //
    const int in1_r,               //
    const int batch_size,          //
    const Activation act_type  //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= out_r * batch_size)
        return;

    const int batch_idx = idx / out_r;
    const int out_idx = idx % out_r;

    const int out_offset = out_idx + batch_idx * out_r;

    float val;
    if(out_idx < in1_r) {
        const int in1_offset = out_idx + batch_idx * in1_r;
        val = in1_v[in1_offset];
    } else {
        const int in2_offset = (out_idx - in1_r) + batch_idx * (out_r - in1_r);
        val = in2_v[in2_offset];
    }

    linear_out[out_offset] = val;
    if(UseActivation)
        activated[out_offset] = activate_fwd(val, act_type);
}

void concat_fwd(                  //
    const DenseMatrix &in1_v,     //
    const DenseMatrix &in2_v,     //
    DenseMatrix &linear_out,      //
    DenseMatrix &activated,       //
    const Activation act_type //
) {
    ASSERT(in1_v.cols() == linear_out.cols() && //
           in2_v.cols() == linear_out.cols() && //
           in1_v.rows() + in2_v.rows() == linear_out.rows());

    ASSERT(in1_v.is_dev_allocated() && //
           in2_v.is_dev_allocated() && //
           linear_out.is_dev_allocated());

    const int blocks = get_num_blocks(linear_out.size(), block_size);

    if(has_activation(act_type)) {
        ASSERT(activated.is_dev_allocated());

        concat_fwd_kernel<true><<<blocks, block_size>>>( //
            in1_v.dev_address(),
            in2_v.dev_address(),
            linear_out.dev_address(),
            activated.dev_address(),
            linear_out.rows(),
            in1_v.rows(),
            linear_out.cols(),
            act_type);
    } else {
        concat_fwd_kernel<false><<<blocks, block_size>>>( //
            in1_v.dev_address(),
            in2_v.dev_address(),
            linear_out.dev_address(),
            activated.dev_address(),
            linear_out.rows(),
            in1_v.rows(),
            linear_out.cols(),
            act_type);
    }
}

} // namespace kernel
