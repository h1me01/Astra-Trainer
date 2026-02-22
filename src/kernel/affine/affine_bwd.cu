#include "affine.h"

namespace kernel {

constexpr float alpha = 1.0f;
constexpr float beta = 1.0f;

constexpr int num_threads = 256;

template <Activation act_type>
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
    __shared__ float shared[num_threads];

    const int neuron_idx = blockIdx.x;
    const int tid = threadIdx.x;

    float sum = 0.0f;
    for (int b = tid; b < c; b += blockDim.x)
        sum += out_g[neuron_idx * c + b];

    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        biases_g[neuron_idx] = shared[0];
}

void affine_bwd(Tensor& weights, Tensor& biases, Tensor& in, Tensor& out, const Activation act_type) {
    const auto& in_d = in.get_data();
    auto& in_g = in.get_grads();

    const auto& out_d = out.get_data();
    const auto& out_g = out.get_grads();

    const auto& weights_d = weights.get_data();
    auto& weights_g = weights.get_grads();

    auto& biases_g = biases.get_grads();

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
    if (act_type != Activation::Linear) {
        const int blocks = cuda::ceil_div(out_g.size(), 4 * num_threads);
        DISPATCH_ACTIVATION(
            act_type,
            activate_bwd_kernel,
            <<<blocks, num_threads>>>(out_d.dev_address(), out_g.dev_address(), out_g.size())
        );
    }

    // update biases gradients
    {
        const int blocks = cuda::ceil_div(out_g.rows(), num_threads);
        biases_bwd_kernel<<<blocks, num_threads>>>(
            biases_g.dev_address(), out_g.dev_address(), out_g.cols(), out_g.rows()
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
        in_d.dev_address(),      // B
        in_d.rows(),             // ldb
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
        weights_d.rows(),        // k
        &alpha,                  // alpha
        weights_d.dev_address(), // A
        weights_d.rows(),        // lda
        out_g.dev_address(),     // B
        out_g.rows(),            // ldb
        &beta,                   // beta
        in_g.dev_address(),      // C
        in_g.rows()              // ldc
    );
}

} // namespace kernel
