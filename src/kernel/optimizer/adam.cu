#include "optimizer.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void adam_kernel( //
    float *vals,             //
    const float *grads,      //
    float *moms,             //
    float *vels,             //
    const float lr,          //
    const float beta1,       //
    const float beta2,       //
    const float eps,         //
    const float decay,       //
    const float min_val,     //
    const float max_val,     //
    const float grad_scale,  //
    const int size           //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    const float grad = grads[idx] * grad_scale;

    float mom = moms[idx];
    float vel = vels[idx];
    float val = vals[idx] * decay;

    mom = beta1 * mom + (1.0f - beta1) * grad;
    vel = beta2 * vel + (1.0f - beta2) * grad * grad;

    val -= lr * mom / (sqrtf(vel) + eps);

    moms[idx] = mom;
    vels[idx] = vel;
    vals[idx] = clamp(val, min_val, max_val);
}

void adam_optim(           //
    Tensor<float> &param,  //
    Array<float> &moms,    //
    Array<float> &vels,    //
    const float lr,        //
    const float beta1,     //
    const float beta2,     //
    const float eps,       //
    const float decay,     //
    const float grad_scale //
) {
    const float min_val = param.lower_bound();
    const float max_val = param.upper_bound();

    auto &vals = param.get_values();
    auto &grads = param.get_gradients();

    ASSERT(moms.size() == vals.size() && vels.size() == vals.size());

    ASSERT(vals.is_dev_allocated() &&  //
           grads.is_dev_allocated() && //
           moms.is_dev_allocated() &&  //
           vels.is_dev_allocated());

    const int blocks = get_num_blocks(vals.size(), block_size);
    adam_kernel<<<blocks, block_size>>>( //
        vals.dev_address(),
        grads.dev_address(),
        moms.dev_address(),
        vels.dev_address(),
        lr,
        beta1,
        beta2,
        eps,
        optimizer_utils::get_decay(lr, decay),
        min_val,
        max_val,
        grad_scale,
        vals.size());

    grads.clear_dev();
}

} // namespace kernel
