#include "loss.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void mse_kernel(
    const float* targets, const float* out, float* grads, float* loss, const int size, const Activation act_type
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;

    if (idx < size) {
        const float act = activate_fwd(out[idx], act_type);
        const float diff = act - targets[idx];
        grads[idx] = 2.0f * diff * activate_bwd(act, act_type);
        val = diff * diff;
    }

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    if ((threadIdx.x % 32) == 0 && val != 0.0f)
        atomicAdd(loss, val);
}

void mse_loss(
    const Array<float>& targets,
    Array<float>& loss,
    const DenseMatrix& out,
    DenseMatrix& grads,
    const Activation act_type
) {
    ASSERT(
        out.is_dev_allocated() &&     //
        grads.is_dev_allocated() &&   //
        targets.is_dev_allocated() && //
        loss.is_dev_allocated()
    );

    const int blocks = get_num_blocks(out.size(), block_size);
    mse_kernel<<<blocks, block_size>>>(
        targets.dev_address(), out.dev_address(), grads.dev_address(), loss.dev_address(), out.size(), act_type
    );
}

} // namespace kernel
