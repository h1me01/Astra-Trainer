#include "optimizer.h"

namespace kernel {

constexpr int block_size = 1024;

constexpr float epsilon = 1e-8f;

__device__ __forceinline__ void adam_update_f4(
    float4& val,
    float4& mom,
    float4& vel,
    const float4& grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float decay,
    const float min_val,
    const float max_val,
    const float grad_scale
) {
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;

#pragma unroll
    for (int i = 0; i < 4; i++) {
        float* v = &val.x + i;
        float* m = &mom.x + i;
        float* ve = &vel.x + i;
        const float* g = &grad.x + i;

        const float scaled_grad = (*g) * grad_scale;
        *v *= decay;
        *m = beta1 * (*m) + one_minus_beta1 * scaled_grad;
        *ve = beta2 * (*ve) + one_minus_beta2 * scaled_grad * scaled_grad;
        *v = clamp(*v - lr * (*m) / (sqrtf(*ve) + epsilon), min_val, max_val);
    }
}

__global__ void adam_kernel(
    float* vals,
    const float* grads,
    float* moms,
    float* vels,
    const float lr,
    const float beta1,
    const float beta2,
    const float decay,
    const float min_val,
    const float max_val,
    const float grad_scale,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= size)
        return;

    if (vec_idx + 4 <= size) {
        float4 val = as_vec<float4>(vals)[idx];
        float4 mom = as_vec<float4>(moms)[idx];
        float4 vel = as_vec<float4>(vels)[idx];
        const float4 grad = as_vec<const float4>(grads)[idx];

        adam_update_f4(val, mom, vel, grad, lr, beta1, beta2, decay, min_val, max_val, grad_scale);

        as_vec<float4>(vals)[idx] = val;
        as_vec<float4>(moms)[idx] = mom;
        as_vec<float4>(vels)[idx] = vel;
    } else {
        const float one_minus_beta1 = 1.0f - beta1;
        const float one_minus_beta2 = 1.0f - beta2;

        for (int i = vec_idx; i < size; i++) {
            const float scaled_grad = grads[i] * grad_scale;
            vals[i] *= decay;
            moms[i] = beta1 * moms[i] + one_minus_beta1 * scaled_grad;
            vels[i] = beta2 * vels[i] + one_minus_beta2 * scaled_grad * scaled_grad;
            vals[i] = clamp(vals[i] - lr * moms[i] / (sqrtf(vels[i]) + epsilon), min_val, max_val);
        }
    }
}

void adam_optim(
    Tensor& param,
    Array<float>& moms,
    Array<float>& vels,
    const float lr,
    const float beta1,
    const float beta2,
    const float decay,
    const float grad_scale
) {
    const float min_val = param.lower_bound();
    const float max_val = param.upper_bound();

    auto& vals = param.get_data();
    auto& grads = param.get_grads();

    ASSERT(moms.size() == vals.size() && vels.size() == vals.size());

    ASSERT(
        vals.is_dev_allocated() &&  //
        grads.is_dev_allocated() && //
        moms.is_dev_allocated() &&  //
        vels.is_dev_allocated()
    );

    const int blocks = get_num_blocks(vals.size(), 4 * block_size);
    adam_kernel<<<blocks, block_size>>>(
        vals.dev_address(),
        grads.dev_address(),
        moms.dev_address(),
        vels.dev_address(),
        lr,
        beta1,
        beta2,
        1.0f - lr * decay,
        min_val,
        max_val,
        grad_scale,
        vals.size()
    );
}

} // namespace kernel
