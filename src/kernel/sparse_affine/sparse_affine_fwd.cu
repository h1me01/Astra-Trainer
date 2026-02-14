#include "sparse_affine.h"

namespace kernel {

constexpr int block_size = 256;

template <Activation act_type>
__global__ void sparse_affine_fwd_kernel(
    const float* weights_v,
    const float* biases_v,
    float* out_d,
    const int* features,
    const int weights_r,
    const int out_r,
    const int batch_size,
    const int max_entries
) {
    const int batch = blockIdx.x;
    if (batch >= batch_size)
        return;

    extern __shared__ int s_features[];
    for (int i = threadIdx.x; i < max_entries; i += blockDim.x)
        s_features[i] = features[batch * max_entries + i];
    __syncthreads();

    const int vec = weights_r / 4;
    const int rem = weights_r % 4;

    const float4* w4 = reinterpret_cast<const float4*>(weights_v);
    const float4* b4 = reinterpret_cast<const float4*>(biases_v);
    float4* o4 = reinterpret_cast<float4*>(out_d);

    for (int k = threadIdx.x; k < vec; k += blockDim.x) {
        float4 sum = b4[k];

#pragma unroll
        for (int i = 0; i < max_entries; i++) {
            int f = s_features[i];
            if (f == -1)
                break;
            add_t4(sum, w4[f * vec + k]);
        }

        o4[(batch * out_r / 4) + k] = activate_fwd_f4<act_type>(sum);
    }

    if (rem) {
        for (int n = (vec * 4) + threadIdx.x; n < weights_r; n += blockDim.x) {
            float sum = biases_v[n];

#pragma unroll
            for (int i = 0; i < max_entries; i++) {
                int f = s_features[i];
                if (f == -1)
                    break;
                sum += weights_v[f * weights_r + n];
            }

            out_d[batch * out_r + n] = activate_fwd<act_type>(sum);
        }
    }
}

void sparse_affine_fwd(
    const DenseMatrix& weights_v,
    const DenseMatrix& biases_v,
    DenseMatrix& out_d,
    const Array<int>& features,
    const int max_entries,
    const int out_offset,
    const Activation act_type
) {
    const bool is_double = out_d.rows() / 2 == weights_v.rows();

    ASSERT(weights_v.rows() == biases_v.rows());
    ASSERT(weights_v.rows() == out_d.rows() / (is_double ? 2 : 1));

    ASSERT(
        weights_v.is_dev_allocated() && //
        biases_v.is_dev_allocated() &&  //
        out_d.is_dev_allocated() &&     //
        features.is_dev_allocated()
    );

    int shared_mem_size = max_entries * sizeof(int);
    DISPATCH_ACTIVATION(
        act_type,
        sparse_affine_fwd_kernel,
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