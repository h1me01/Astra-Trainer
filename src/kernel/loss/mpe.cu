#include "loss.h"

namespace kernel {

constexpr int BLOCK_SIZE = 1024;

template <typename Op>
__global__ void mpe_kernel(
    const float* targets,
    const float* out_d,
    float* out_g,
    float* loss,
    const float power,
    const float norm_factor,
    const int size,
    Op op
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_loss = 0.0f;

    if (idx < size) {
        const float act = op.forward(out_d[idx]);
        const float diff = act - targets[idx];
        const float abs_diff = abs(diff);
        const float sign = (diff > 0.0f) ? 1.0f : -1.0f;

        out_g[idx] = sign * power * powf(abs_diff, power - 1.0f) * op.backward<true>(act) * norm_factor;
        local_loss = powf(abs_diff, power) * norm_factor;
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = local_loss;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(loss, smem[0]);
}

void mpe_loss(const Array<float>& targets, Array<float>& loss, Tensor& out, const float power, ActOp op) {
    const auto& out_d = out.data();
    auto& out_g = out.grad();

    CHECK(
        out_d.is_dev_allocated() &&   //
        out_g.is_dev_allocated() &&   //
        targets.is_dev_allocated() && //
        loss.is_dev_allocated()
    );

    const float norm_factor = 1.0f / out_d.size();

    const int blocks = cuda::ceil_div(out_d.size(), BLOCK_SIZE);

    std::visit(
        [&](auto op) {
            mpe_kernel<<<blocks, BLOCK_SIZE>>>(
                targets.dev_address(),
                out_d.dev_address(),
                out_g.dev_address(),
                loss.dev_address(),
                power,
                norm_factor,
                out_d.size(),
                op
            );
        },
        op
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kernel
