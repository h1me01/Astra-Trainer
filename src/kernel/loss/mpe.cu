#include "loss.h"

namespace kernel {

constexpr int BLOCK_SIZE = 1024;

template <typename Op>
__global__ void mpe_kernel(
    const float* targets, const float* out_d, float* out_g, float* loss, const float power, const int size, Op op
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_loss = 0.0f;

    if (idx < size) {
        const float act = op.forward(out_d[idx]);
        const float diff = act - targets[idx];
        const float abs_diff = abs(diff);
        const float sign = (diff > 0.0f) ? 1.0f : -1.0f;

        out_g[idx] = sign * power * powf(abs_diff, power - 1.0f) * op.backward<true>(act);
        local_loss = powf(abs_diff, power);
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

template <typename Op>
void mpe_loss(const Array<float>& targets, Array<float>& loss, Tensor& out, const float power, Op op) {
    const auto& out_d = out.data();
    auto& out_g = out.grad();

    CHECK(
        out_d.is_dev_allocated() &&   //
        out_g.is_dev_allocated() &&   //
        targets.is_dev_allocated() && //
        loss.is_dev_allocated()
    );

    const int blocks = cuda::ceil_div(out_d.size(), BLOCK_SIZE);
    mpe_kernel<<<blocks, BLOCK_SIZE>>>(
        targets.dev_address(), out_d.dev_address(), out_g.dev_address(), loss.dev_address(), power, out_d.size(), op
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

void mpe_loss(const Array<float>& targets, Array<float>& loss, Tensor& out, const float power, ActOp op) {
    std::visit([&](const auto& concrete_op) { mpe_loss(targets, loss, out, power, concrete_op); }, op);
}

} // namespace kernel
