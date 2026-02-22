#include "loss.h"

namespace kernel {

constexpr int num_threads = 1024;

template <Activation act_type>
__global__ void
mpe_kernel(const float* targets, const float* out_d, float* out_g, float* loss, const float power, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    const float act = activate_fwd<act_type>(out_d[idx]);
    const float diff = act - targets[idx];
    const float abs_diff = abs(diff);
    const float sign = (diff > 0.0f) ? 1.0f : -1.0f;

    out_g[idx] = sign * power * powf(abs_diff, power - 1.0f) * activate_bwd<act_type, true>(act);
    atomicAdd(loss, powf(abs_diff, power));
}

void mpe_loss(
    const Array<float>& targets, Array<float>& loss, Tensor& out, const float power, const Activation act_type
) {
    const auto& out_d = out.get_data();
    auto& out_g = out.get_grads();

    CHECK(
        out_d.is_dev_allocated() &&   //
        out_g.is_dev_allocated() &&   //
        targets.is_dev_allocated() && //
        loss.is_dev_allocated()
    );

    const int blocks = cuda::ceil_div(out_d.size(), num_threads);
    DISPATCH_ACTIVATION(
        act_type,
        mpe_kernel,
        <<<blocks, num_threads>>>(
            targets.dev_address(), out_d.dev_address(), out_g.dev_address(), loss.dev_address(), power, out_d.size()
        )
    );
}

} // namespace kernel
