#include "optimizer.h"

namespace kernel {

using namespace optim_utils;

constexpr int block_size = 1024;

// https://github.com/LiyuanLucasLiu/RAdam.git
__global__ void radam_kernel(
    float* vals,
    const float* grads,
    float* moms,
    float* vels,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float decay,
    const float min_val,
    const float max_val,
    const float grad_scale,
    const RAdamParams radam_params,
    const int step,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    const float grad = grads[idx] * grad_scale;

    float mom = moms[idx];
    float vel = vels[idx];
    float val = vals[idx] * decay;

    mom = beta1 * mom + (1.0f - beta1) * grad;
    vel = beta2 * vel + (1.0f - beta2) * grad * grad;

    val -= get_radam_update(mom, vel, lr, eps, radam_params);

    moms[idx] = mom;
    vels[idx] = vel;
    vals[idx] = clamp(val, min_val, max_val);
}

void radam_optim(
    Tensor& param,
    Array<float>& moms,
    Array<float>& vels,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float decay,
    const float grad_scale,
    const RAdamParams radam_params,
    const int step
) {
    auto& vals = param.get_data();
    auto& grads = param.get_grads();

    ASSERT(moms.size() == vals.size() && vels.size() == vals.size());

    ASSERT(
        vals.is_dev_allocated() &&  //
        grads.is_dev_allocated() && //
        moms.is_dev_allocated() &&  //
        vels.is_dev_allocated()
    );

    const int blocks = get_num_blocks(vals.size(), block_size);
    radam_kernel<<<blocks, block_size>>>(
        vals.dev_address(),
        grads.dev_address(),
        moms.dev_address(),
        vels.dev_address(),
        lr,
        beta1,
        beta2,
        eps,
        get_decay(lr, decay),
        param.lower_bound(),
        param.upper_bound(),
        grad_scale,
        radam_params,
        step,
        vals.size()
    );

    grads.clear_dev();
}

} // namespace kernel
