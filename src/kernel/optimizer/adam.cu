#include "optimizer.h"

namespace kernel {

constexpr int block_size = 1024;

constexpr float epsilon = 1e-8f;

__device__ __forceinline__ void update_mom_f4(float4& mom, const float4& grad, const float beta1) {
    const float one_minus_beta1 = 1.0f - beta1;
    mom.x = beta1 * mom.x + one_minus_beta1 * grad.x;
    mom.y = beta1 * mom.y + one_minus_beta1 * grad.y;
    mom.z = beta1 * mom.z + one_minus_beta1 * grad.z;
    mom.w = beta1 * mom.w + one_minus_beta1 * grad.w;
}

__device__ __forceinline__ void update_vel_f4(float4& vel, const float4& grad, const float beta2) {
    const float one_minus_beta2 = 1.0f - beta2;
    vel.x = beta2 * vel.x + one_minus_beta2 * grad.x * grad.x;
    vel.y = beta2 * vel.y + one_minus_beta2 * grad.y * grad.y;
    vel.z = beta2 * vel.z + one_minus_beta2 * grad.z * grad.z;
    vel.w = beta2 * vel.w + one_minus_beta2 * grad.w * grad.w;
}

__device__ __forceinline__ void clamp_f4(float4& val, const float min_val, const float max_val) {
    val.x = clamp(val.x, min_val, max_val);
    val.y = clamp(val.y, min_val, max_val);
    val.z = clamp(val.z, min_val, max_val);
    val.w = clamp(val.w, min_val, max_val);
}

__device__ __forceinline__ void adam_update_f4(float4& val, const float4& mom, const float4& vel, const float lr) {
    val.x -= lr * mom.x / (sqrtf(vel.x) + epsilon);
    val.y -= lr * mom.y / (sqrtf(vel.y) + epsilon);
    val.z -= lr * mom.z / (sqrtf(vel.z) + epsilon);
    val.w -= lr * mom.w / (sqrtf(vel.w) + epsilon);
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

    const int remaining = min(4, size - vec_idx);
    if (remaining == 4) {
        float4* vals_v4 = (float4*)vals;
        float4* moms_v4 = (float4*)moms;
        float4* vels_v4 = (float4*)vels;
        const float4* grads_v4 = (const float4*)grads;

        float4 grad = grads_v4[idx];
        float4 val = vals_v4[idx];
        float4 mom = moms_v4[idx];
        float4 vel = vels_v4[idx];

        mul_t4(grad, grad_scale);
        mul_t4(val, decay);

        update_mom_f4(mom, grad, beta1);
        update_vel_f4(vel, grad, beta2);
        adam_update_f4(val, mom, vel, lr);
        clamp_f4(val, min_val, max_val);

        vals_v4[idx] = val;
        moms_v4[idx] = mom;
        vels_v4[idx] = vel;
    } else {
        for (int i = vec_idx; i < vec_idx + remaining; i++) {
            const float grad = grads[i] * grad_scale;
            float mom = moms[i];
            float vel = vels[i];
            float val = vals[i] * decay;

            mom = beta1 * mom + (1.0f - beta1) * grad;
            vel = beta2 * vel + (1.0f - beta2) * grad * grad;
            val -= lr * mom / (sqrtf(vel) + epsilon);

            moms[i] = mom;
            vels[i] = vel;
            vals[i] = clamp(val, min_val, max_val);
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

    const int blocks = (vals.size() + block_size * 4 - 1) / (block_size * 4);
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
