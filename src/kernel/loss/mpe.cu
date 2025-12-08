#include "loss.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void mpe_kernel(       //
    const float *targets,         //
    const float *out,             //
    float *grads,                 //
    float *loss,                  //
    const float power,            //
    const int size,               //
    const Activation act_type //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    const float pred = out[idx];
    const float act = activate_fwd(pred, act_type);
    const float diff = act - targets[idx];
    const float abs_diff = fabsf(diff);

    const float grad_magnitude = power * powf(abs_diff, power - 1.0f);
    const float sign_diff = (diff > 0.0f) ? 1.0f : -1.0f;
    grads[idx] = grad_magnitude * sign_diff * activate_bwd(pred, act_type);

    float p = powf(abs_diff, power);
    if(p != 0.0f)
        atomicAdd(&loss[0], p);
}

void mpe_loss(                    //
    const Array<float> &targets,  //
    Array<float> &loss,           //
    const DenseMatrix &out,       //
    DenseMatrix &grads,           //
    const float power,            //
    const Activation act_type //
) {
    ASSERT(out.is_dev_allocated() &&     //
           grads.is_dev_allocated() &&   //
           targets.is_dev_allocated() && //
           loss.is_dev_allocated());

    const int blocks = get_num_blocks(out.size(), block_size);
    mpe_kernel<<<blocks, block_size>>>( //
        targets.dev_address(),
        out.dev_address(),
        grads.dev_address(),
        loss.dev_address(),
        power,
        out.size(),
        act_type);
}

} // namespace kernel
