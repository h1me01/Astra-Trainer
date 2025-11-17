#include "activation.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void activate_fwd_kernel( //
    const float *in_v,               //
    float *out_v,                    //
    const int size,                  //
    const ActivationType type        //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        out_v[idx] = activate(in_v[idx], type);
}

void activate_fwd(            //
    const DenseMatrix &in_v,  //
    DenseMatrix &out_v,       //
    const ActivationType type //
) {
    ASSERT(in_v.size() == out_v.size());

    ASSERT(in_v.is_dev_allocated() && out_v.is_dev_allocated());

    const int blocks = get_num_blocks(in_v.size(), block_size);
    activate_fwd_kernel<<<blocks, block_size>>>( //
        in_v.dev_address(),
        out_v.dev_address(),
        in_v.size(),
        type);
}

} // namespace kernel
