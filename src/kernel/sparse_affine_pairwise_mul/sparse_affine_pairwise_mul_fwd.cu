#include "sparse_affine_pairwise_mul.h"

namespace kernel {

constexpr int block_size = 256;

template <Activation act_type>
__global__ void sparse_affine_pairwise_mul_fwd_kernel(
    const float* weights_v,
    const float* biases_v,
    float* out_d,
    const int* features,
    const int weights_r,
    const int out_r,
    const int batch_size,
    const int max_entries
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size)
        return;

    extern __shared__ int s_features[];
    for (int i = threadIdx.x; i < max_entries; i += blockDim.x)
        s_features[i] = features[batch_idx * max_entries + i];
    __syncthreads();

    const int half = weights_r / 2;
    const int half4 = half / 4;
    const int rem = half % 4;

    const float4* w4 = reinterpret_cast<const float4*>(weights_v);
    const float4* b4 = reinterpret_cast<const float4*>(biases_v);
    float4* o4 = reinterpret_cast<float4*>(out_d);

    const int col_stride4 = weights_r / 4;

    for (int k = threadIdx.x; k < half4; k += blockDim.x) {
        float4 sum_a = b4[k];
        float4 sum_b = b4[half4 + k];

#pragma unroll
        for (int i = 0; i < max_entries; i++) {
            int f = s_features[i];
            if (f == -1)
                break;
            int base = f * col_stride4;
            add_t4(sum_a, w4[base + k]);
            add_t4(sum_b, w4[base + half4 + k]);
        }

        activate_fwd_f4<act_type>(sum_a);
        activate_fwd_f4<act_type>(sum_b);

        float4 r;
        r.x = sum_a.x * sum_b.x;
        r.y = sum_a.y * sum_b.y;
        r.z = sum_a.z * sum_b.z;
        r.w = sum_a.w * sum_b.w;

        int out_idx4 = (batch_idx * out_r / 4) + k;
        o4[out_idx4] = r;
    }

    if (rem) {
        for (int n = (half4 * 4) + threadIdx.x; n < half; n += blockDim.x) {
            float sum_a = biases_v[n];
            float sum_b = biases_v[n + half];

#pragma unroll
            for (int i = 0; i < max_entries; i++) {
                int f = s_features[i];
                if (f == -1)
                    break;
                int base = f * weights_r;
                sum_a += weights_v[base + n];
                sum_b += weights_v[base + n + half];
            }

            out_d[batch_idx * out_r + n] = activate_fwd<act_type>(sum_a) * activate_fwd<act_type>(sum_b);
        }
    }
}

void sparse_affine_pairwise_mul_fwd(
    const DenseMatrix& weights_v,
    const DenseMatrix& biases_v,
    DenseMatrix& out_d,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
) {
    ASSERT(weights_v.rows() == biases_v.rows());

    ASSERT(
        weights_v.is_dev_allocated() && //
        biases_v.is_dev_allocated() &&  //
        out_d.is_dev_allocated() &&     //
        features.is_dev_allocated()
    );

    int shared_mem_size = max_entries * sizeof(int);
    DISPATCH_ACTIVATION(
        act_type,
        sparse_affine_pairwise_mul_fwd_kernel,
        <<<out_d.cols(), block_size, shared_mem_size>>>(
            weights_v.dev_address(),
            biases_v.dev_address(),
            out_d.dev_address() + out_offset,
            features.dev_address(),
            weights_v.rows(),
            out_d.rows(),
            out_d.cols(),
            max_entries
        )
    );
}

} // namespace kernel
