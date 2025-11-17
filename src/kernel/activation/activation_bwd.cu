#include "activation.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void activate_bwd_kernel( //
    const float *in_v,               //
    float *in_g,                     //
    const float *out_g,              //
    const int size,                  //
    const ActivationType type        //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        in_g[idx] = activate_der(in_v[idx], type) * out_g[idx];
}

void activate_bwd(            //
    Tensor &in,               //
    const DenseMatrix &out_g, //
    const ActivationType type //
) {
    const auto &in_v = in.get_values();
    auto &in_g = in.get_gradients();

    ASSERT(in_v.size() == out_g.size());

    ASSERT(in_v.is_dev_allocated() && //
           in_g.is_dev_allocated() && //
           out_g.is_dev_allocated());

    const int blocks = get_num_blocks(in_v.size(), block_size);
    activate_bwd_kernel<<<blocks, block_size>>>( //
        in_v.dev_address(),
        in_g.dev_address(),
        out_g.dev_address(),
        in_v.size(),
        type);
}

} // namespace kernel
