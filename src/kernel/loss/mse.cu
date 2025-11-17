#include "loss.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void mse_kernel(        //
    const float *targets,          //
    const float *out_v,            //
    float *out_g,                  //
    float *loss,                   //
    const ActivationType act_type, //
    const int size                 //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    const float pred = out_v[idx];
    const float act = activate(pred, act_type);
    const float diff = act - targets[idx];

    out_g[idx] = 2.0f * diff * activate_der(pred, act_type);

    float sq = diff * diff;
    if(sq != 0.0f)
        atomicAdd(loss, sq);
}

void mse_loss(                    //
    const Array<float> &targets,  //
    Array<float> &loss,           //
    Tensor &out,                  //
    const ActivationType act_type //
) {
    const auto &out_v = out.get_values();
    auto &out_g = out.get_gradients();

    ASSERT(out_v.is_dev_allocated() &&   //
           out_g.is_dev_allocated() &&   //
           targets.is_dev_allocated() && //
           loss.is_dev_allocated());

    const int blocks = get_num_blocks(out_v.size(), block_size);
    mse_kernel<<<blocks, block_size>>>( //
        targets.dev_address(),
        out_v.dev_address(),
        out_g.dev_address(),
        loss.dev_address(),
        act_type,
        out_v.size());
}

} // namespace kernel
