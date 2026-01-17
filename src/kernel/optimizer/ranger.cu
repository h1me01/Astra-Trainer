#include "optimizer.h"

namespace kernel {

constexpr int block_size = 1024;

// https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer.git
__global__ void ranger_kernel(
    float* vals,
    const float* grads,
    float* moms,
    float* vels,
    float* slow_buffer,
    const float lr,
    const float beta1,
    const float beta2,
    const float beta1_t,
    const float beta2_t,
    const float eps,
    const float decay,
    const float min_val,
    const float max_val,
    const float grad_scale,
    const int N_sma,
    const int N_sma_max,
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

    val -= optimizer_utils::get_radam_update(mom, vel, lr, eps, beta1_t, beta2_t, N_sma, N_sma_max);

    // lookahead
    if (step % 6 == 0) {
        float slow_val = slow_buffer[idx];
        slow_val += 0.5 * (val - slow_val);
        val = slow_val;
        slow_buffer[idx] = slow_val;
    }

    moms[idx] = mom;
    vels[idx] = vel;
    vals[idx] = clamp(val, min_val, max_val);
}

void ranger_optim(
    Tensor& param,
    Array<float>& moms,
    Array<float>& vels,
    Array<float>& slow_buffer,
    const float lr,
    const float beta1,
    const float beta2,
    const float beta1_t,
    const float beta2_t,
    const float eps,
    const float decay,
    const float grad_scale,
    const int N_sma,
    const int N_sma_max,
    const int step
) {
    const float min_val = param.lower_bound();
    const float max_val = param.upper_bound();

    auto& vals = param.get_values();
    auto& grads = param.get_gradients();

    ASSERT(
        moms.size() == vals.size() && //
        vels.size() == vals.size() && //
        slow_buffer.size() == vals.size()
    );

    ASSERT(
        vals.is_dev_allocated() &&  //
        grads.is_dev_allocated() && //
        moms.is_dev_allocated() &&  //
        vels.is_dev_allocated() &&  //
        slow_buffer.is_dev_allocated()
    );

    const int blocks = get_num_blocks(vals.size(), block_size);
    ranger_kernel<<<blocks, block_size>>>(
        vals.dev_address(),
        grads.dev_address(),
        moms.dev_address(),
        vels.dev_address(),
        slow_buffer.dev_address(),
        lr,
        beta1,
        beta2,
        beta1_t,
        beta2_t,
        eps,
        optimizer_utils::get_decay(lr, decay),
        min_val,
        max_val,
        grad_scale,
        N_sma,
        N_sma_max,
        step,
        vals.size()
    );

    grads.clear_dev();
}

} // namespace kernel
