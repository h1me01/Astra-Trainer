#include "affine.h"

namespace kernel {

constexpr float alpha = 1.0f;
constexpr float beta = 1.0f;

constexpr int num_threads = 256;

template <ActivationType act_type>
__global__ void activate_bwd_kernel(const float* out_d, float* out_g, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= size)
        return;

    if (vec_idx + 4 <= size) {
        float4 v = as_vec<const float4>(out_d)[idx];
        float4 g = as_vec<float4>(out_g)[idx];

        g.x *= activate_bwd<act_type, true>(v.x);
        g.y *= activate_bwd<act_type, true>(v.y);
        g.z *= activate_bwd<act_type, true>(v.z);
        g.w *= activate_bwd<act_type, true>(v.w);

        as_vec<float4>(out_g)[idx] = g;
    } else {
        for (int i = vec_idx; i < size; i++)
            out_g[i] *= activate_bwd<act_type, true>(out_d[i]);
    }
}

__global__ void biases_bwd_kernel(float* biases_g, const float* out_g, const int r, const int c) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= r * c)
        return;

    const float grad = out_g[idx];
    if (grad != 0)
        atomicAdd(&biases_g[idx % r], grad);
}

void affine_bwd(Tensor& weights, Tensor& biases, Tensor& in, Tensor& out, const ActivationType act_type) {
    const auto& in_d = in.data();
    auto& in_g = in.grads();

    const auto& out_d = out.data();
    const auto& out_g = out.grads();

    const auto& weights_d = weights.data();
    auto& weights_g = weights.grads();

    auto& biases_g = biases.grads();

    CHECK(
        biases_g.cols() == 1 &&             //
        out_g.rows() == biases_g.rows() &&  //
        in_g.cols() == out_g.cols() &&      //
        weights_g.rows() == out_g.rows() && //
        weights_g.cols() == in_g.rows()
    );

    CHECK(
        weights_d.is_dev_allocated() && //
        weights_g.is_dev_allocated() && //
        biases_g.is_dev_allocated() &&  //
        in_d.is_dev_allocated() &&      //
        in_g.is_dev_allocated() &&      //
        out_g.is_dev_allocated()
    );

    // update gradients if activation was used
    if (act_type != ActivationType::Linear) {
        const int blocks = cuda::ceil_div(out_g.size(), 4 * num_threads);
        DISPATCH_ACTIVATION(
            act_type,
            activate_bwd_kernel,
            <<<blocks, num_threads>>>(out_d.dev_address(), out_g.dev_address(), out_g.size())
        );

        CUDA_KERNEL_LAUNCH_CHECK();
    }

    // update biases gradients
    {
        const int blocks = cuda::ceil_div(out_g.size(), num_threads);
        biases_bwd_kernel<<<blocks, num_threads>>>(
            biases_g.dev_address(), out_g.dev_address(), out_g.rows(), out_g.cols()
        );

        CUDA_KERNEL_LAUNCH_CHECK();
    }

    // update weights gradient
    cublas::sgemm(
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        weights_g.rows(),
        weights_g.cols(),
        out_g.cols(),
        alpha,
        out_g.dev_address(),
        out_g.rows(),
        in_d.dev_address(),
        in_d.rows(),
        beta,
        weights_g.dev_address(),
        weights_g.rows()
    );

    // calculates delta for the layer before this one as well
    cublas::sgemm(
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        in_g.rows(),
        in_g.cols(),
        weights_d.rows(),
        alpha,
        weights_d.dev_address(),
        weights_d.rows(),
        out_g.dev_address(),
        out_g.rows(),
        beta,
        in_g.dev_address(),
        in_g.rows()
    );
}

} // namespace kernel
