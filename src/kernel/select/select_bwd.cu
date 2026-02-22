#include "select.h"

namespace kernel {

constexpr int num_threads = 256;

template <Activation act_type>
__global__ void select_bwd_kernel(
    float* in_g,
    const float* out_d,
    const float* out_g,
    const int* indices,
    const int in_r,
    const int out_r,
    const int batch_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_r)
        return;

    const int batch_idx = idx / out_r;
    const int out_idx = idx % out_r;

    const int bucket = indices[batch_idx];
    const int in_offset = in_r * batch_idx + out_r * bucket + out_idx;

    const int out_offset = out_r * batch_idx + out_idx;
    in_g[in_offset] += out_g[out_offset] * activate_bwd<act_type, true>(out_d[out_offset]);
}

void select_bwd(DenseMatrix& in_g, const Tensor& out, const Array<int>& indices, const Activation act_type) {
    auto& out_d = out.get_data();
    auto& out_g = out.get_grads();

    CHECK(in_g.cols() == out_g.cols() && out_g.cols() == indices.size());

    CHECK(
        in_g.is_dev_allocated() &&  //
        out_d.is_dev_allocated() && //
        out_g.is_dev_allocated()    //
        && indices.is_dev_allocated()
    );

    const int blocks = cuda::ceil_div(out_g.size(), num_threads);
    DISPATCH_ACTIVATION(
        act_type,
        select_bwd_kernel,
        <<<blocks, num_threads>>>(
            in_g.dev_address(),
            out_d.dev_address(),
            out_g.dev_address(),
            indices.dev_address(),
            in_g.rows(),
            out_g.rows(),
            out_g.cols()
        )
    );
}

} // namespace kernel
