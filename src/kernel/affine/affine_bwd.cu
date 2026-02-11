#include "affine.h"

namespace kernel {

constexpr float alpha = 1.0f;
constexpr float beta = 1.0f;
constexpr float beta_zero = 0.0f;

constexpr int block_size = 256;

template <Activation act_type>
__global__ void activate_bwd_kernel(const float* out_d, float* out_g, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= size)
        return;

    const int remaining = min(4, size - vec_idx);
    if (remaining == 4) {
        float4 v = ((const float4*)out_d)[idx];
        float4 g = ((float4*)out_g)[idx];

        g.x *= activate_bwd<act_type>(v.x);
        g.y *= activate_bwd<act_type>(v.y);
        g.z *= activate_bwd<act_type>(v.z);
        g.w *= activate_bwd<act_type>(v.w);

        ((float4*)out_g)[idx] = g;
    } else {
        for (int i = vec_idx; i < vec_idx + remaining; i++)
            out_g[i] *= activate_bwd<act_type>(out_d[i]);
    }
}

__global__ void biases_bwd_kernel(float* biases_g, const float* out_g, const int r, const int c) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= r * c)
        return;

    const int neuron_idx = idx % r;
    atomicAdd(&biases_g[neuron_idx], out_g[idx]);
}

void affine_bwd(Tensor& weights, Tensor& biases, Tensor& in, Tensor& out, Activation act_type) {
    const auto& in_v = in.get_data();
    auto& in_g = in.get_grads();

    const auto& out_d = out.get_data();
    const auto& out_g = out.get_grads();

    const auto& weights_v = weights.get_data();
    auto& weights_g = weights.get_grads();

    auto& biases_g = biases.get_grads();

    ASSERT(
        biases_g.cols() == 1 &&             //
        out_g.rows() == biases_g.rows() &&  //
        in_g.cols() == out_g.cols() &&      //
        weights_g.rows() == out_g.rows() && //
        weights_g.cols() == in_g.rows()
    );

    ASSERT(
        weights_v.is_dev_allocated() && //
        weights_g.is_dev_allocated() && //
        biases_g.is_dev_allocated() &&  //
        in_v.is_dev_allocated() &&      //
        in_g.is_dev_allocated() &&      //
        out_g.is_dev_allocated()
    );

    // update gradients if activation was used
    if (act_type != Activation::Linear) {
        const int blocks = get_num_blocks((out_g.size() + 3) / 4, block_size);
        DISPATCH_ACTIVATION(
            act_type,
            activate_bwd_kernel,
            <<<blocks, block_size>>>(out_d.dev_address(), out_g.dev_address(), out_g.size())
        );
    }

    // update biases gradients
    {
        const int blocks = get_num_blocks(out_g.size(), block_size);
        biases_bwd_kernel<<<blocks, block_size>>>(
            biases_g.dev_address(), out_g.dev_address(), out_g.rows(), out_g.cols()
        );
    }

    // update weights gradient
    cublasSgemm(
        CUBLAS_HANDLE,           // handle
        CUBLAS_OP_N,             // transa
        CUBLAS_OP_T,             // transb
        weights_g.rows(),        // m
        weights_g.cols(),        // n
        out_g.cols(),            // k
        &alpha,                  // alpha
        out_g.dev_address(),     // A
        out_g.rows(),            // lda
        in_v.dev_address(),      // B
        in_v.rows(),             // ldb
        &beta,                   // beta
        weights_g.dev_address(), // C
        weights_g.rows()         // ldc
    );

    // calculates delta for the layer before this one as well
    cublasSgemm(
        CUBLAS_HANDLE,           // handle
        CUBLAS_OP_T,             // transa
        CUBLAS_OP_N,             // transb
        in_g.rows(),             // m
        in_g.cols(),             // n
        weights_v.rows(),        // k
        &alpha,                  // alpha
        weights_v.dev_address(), // A
        weights_v.rows(),        // lda
        out_g.dev_address(),     // B
        out_g.rows(),            // ldb
        &beta_zero,              // beta
        in_g.dev_address(),      // C
        in_g.rows()              // ldc
    );
}

} // namespace kernel
