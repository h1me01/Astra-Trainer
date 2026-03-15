#include "optimizer.h"

namespace kernel {

constexpr int BLOCK_SIZE = 1024;

constexpr float epsilon = 1e-8f;

__device__ void adam_update(
    float& val,
    float& mom,
    float& vel,
    const float grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float decay,
    const float min_val,
    const float max_val,
    const float grad_scale
) {
    const float g = grad * grad_scale;
    val *= decay;
    mom = beta1 * mom + (1.0f - beta1) * g;
    vel = beta2 * vel + (1.0f - beta2) * g * g;
    val -= lr * mom / (sqrtf(vel) + epsilon);
    val = clamp(val, min_val, max_val);
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

        const auto update = [&](float& v, float& m, float& ve, float g) {
            adam_update(v, m, ve, g, lr, beta1, beta2, decay, min_val, max_val, grad_scale);
        };

        update(val.x, mom.x, vel.x, grad.x);
        update(val.y, mom.y, vel.y, grad.y);
        update(val.z, mom.z, vel.z, grad.z);
        update(val.w, mom.w, vel.w, grad.w);

        as_vec<float4>(vals)[idx] = val;
        as_vec<float4>(moms)[idx] = mom;
        as_vec<float4>(vels)[idx] = vel;
    } else {
        for (int i = vec_idx; i < size; i++)
            adam_update(vals[i], moms[i], vels[i], grads[i], lr, beta1, beta2, decay, min_val, max_val, grad_scale);
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

    auto& data = param.data();
    auto& grad = param.grad();

    CHECK(moms.size() == data.size() && vels.size() == data.size());

    CHECK(
        data.is_dev_allocated() && //
        grad.is_dev_allocated() && //
        moms.is_dev_allocated() && //
        vels.is_dev_allocated()
    );

    const int blocks = cuda::ceil_div(data.size(), 4 * BLOCK_SIZE);
    adam_kernel<<<blocks, BLOCK_SIZE>>>(
        data.dev_address(),
        grad.dev_address(),
        moms.dev_address(),
        vels.dev_address(),
        lr,
        beta1,
        beta2,
        1.0f - lr * decay,
        min_val,
        max_val,
        grad_scale,
        data.size()
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kernel
