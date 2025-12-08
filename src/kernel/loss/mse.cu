#include "loss.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void mse_kernel(       //
    const float *targets,         //
    const float *out,             //
    float *grads,                 //
    float *loss,                  //
    const int size,               //
    const Activation act_type //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    const float pred = out[idx];
    const float act = activate_fwd(pred, act_type);
    const float diff = act - targets[idx];

    grads[idx] = 2.0f * diff * activate_bwd(pred, act_type);

    float sq = diff * diff;
    if(sq != 0.0f)
        atomicAdd(&loss[0], sq);
}

void mse_loss(                    //
    const Array<float> &targets,  //
    Array<float> &loss,           //
    const DenseMatrix &out,       //
    DenseMatrix &grads,           //
    const Activation act_type //
) {
    ASSERT(out.is_dev_allocated() &&     //
           grads.is_dev_allocated() &&   //
           targets.is_dev_allocated() && //
           loss.is_dev_allocated());

    const int blocks = get_num_blocks(out.size(), block_size);
    mse_kernel<<<blocks, block_size>>>( //
        targets.dev_address(),
        out.dev_address(),
        grads.dev_address(),
        loss.dev_address(),
        out.size(),
        act_type);
}

} // namespace kernel
