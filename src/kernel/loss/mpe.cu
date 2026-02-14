#include "loss.h"

namespace kernel {

constexpr int block_size = 1024;

template <Activation act_type>
__global__ void
mpe_kernel(const float* targets, const float* out_d, float* out_g, float* loss, const float power, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;

    if (idx < size) {
        const float act = activate_fwd<act_type>(out_d[idx]);
        const float diff = act - targets[idx];
        const float abs_diff = fabsf(diff);

        const float p = powf(abs_diff, power);

        const float grad_mag = (abs_diff > 1e-9f) ? (power * p / abs_diff) : 0.0f;
        const float sign = (diff > 0.0f) ? 1.0f : -1.0f;

        out_g[idx] = grad_mag * sign * activate_bwd<act_type, true>(act);
        val = p;
    }

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    if ((threadIdx.x % 32) == 0 && val != 0.0f)
        atomicAdd(loss, val);
}

void mpe_loss(
    const Array<float>& targets, Array<float>& loss, Tensor& out, const float power, const Activation act_type
) {
    const auto& out_d = out.get_data();
    auto& out_g = out.get_grads();

    ASSERT(
        out_d.is_dev_allocated() &&   //
        out_g.is_dev_allocated() &&   //
        targets.is_dev_allocated() && //
        loss.is_dev_allocated()
    );

    const int blocks = get_num_blocks(out_d.size(), block_size);
    DISPATCH_ACTIVATION(
        act_type,
        mpe_kernel,
        <<<blocks, block_size>>>(
            targets.dev_address(), out_d.dev_address(), out_g.dev_address(), loss.dev_address(), power, out_d.size()
        )
    );
}

} // namespace kernel
