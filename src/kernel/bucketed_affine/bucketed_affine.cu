#include "bucketed_affine.h"

// AFFINE

__global__ void bucketed_affine_kernel( //
    const float *input_v,               // [batch_size, input_size]
    const float *weights_v,             // [bucket_size, neuron_size, input_size]
    const float *biases_v,              // [bucket_size, neuron_size]
    float *activated_v,                 // [batch_size, neuron_size]
    float *pre_activated,               // [batch_size, neuron_size]
    const int *bucket_indices,          // [batch_size]
    int batch_size,
    int input_size,
    int neuron_size,
    int bucket_size,
    ActivationType act_type //
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(batch_idx >= batch_size || neuron_idx >= neuron_size)
        return;

    const int bucket_idx = bucket_indices[batch_idx];
    const int bucket_offset = bucket_idx * neuron_size * input_size;

    // compute dot product
    float sum = 0.0f;
    for(int i = 0; i < input_size; ++i) {
        int input_offset = batch_idx * input_size + i;
        int weight_offset = bucket_offset + neuron_idx * input_size + i;
        sum += input_v[input_offset] * weights_v[weight_offset];
    }

    float weighted_sum = sum + biases_v[neuron_idx * bucket_size + bucket_idx];

    int output_idx = batch_idx * neuron_size + neuron_idx;
    pre_activated[output_idx] = weighted_sum;
    activated_v[output_idx] = activate(weighted_sum, act_type);
}

void bucketed_affine_fwd( //
    DenseMatrix &weights_v,
    DenseMatrix &biases_v,
    DenseMatrix &inputs_v,
    DenseMatrix &activated_v,
    DenseMatrix &pre_activated,
    const Array<int> &bucket_indices,
    const ActivationType act_type,
    const int bucket_size //
) {
    const int batch_size = inputs_v.numRows();
    const int input_size = inputs_v.numCols();
    const int neuron_size = weights_v.numRows() / bucket_size;

    ASSERT(activated_v.numCols() == neuron_size && biases_v.numCols() == 1);

    ASSERT(weights_v.numCols() == input_size &&    //
           neuron_size == activated_v.numCols() && //
           batch_size == activated_v.numRows());

    ASSERT(weights_v.devAddress() &&   //
           biases_v.devAddress() &&    //
           inputs_v.devAddress() &&    //
           activated_v.devAddress() && //
           pre_activated.devAddress());

    dim3 block_size(16, 16);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x, //
                   (neuron_size + block_size.y - 1) / block_size.y);

    bucketed_affine_kernel<<<grid_size, block_size>>>( //
        inputs_v.devAddress(),
        weights_v.devAddress(),
        biases_v.devAddress(),
        activated_v.devAddress(),
        pre_activated.devAddress(),
        bucket_indices.devAddress(),
        batch_size,
        input_size,
        neuron_size,
        bucket_size,
        act_type);

    cudaDeviceSynchronize();
}

// AFFINE BW

__global__ void apply_activation_derivative_kernel( //
    float *activated_g,                             // [batch_size, neuron_size]
    const float *pre_activated,                     // [batch_size, neuron_size]
    int batch_size,
    int neuron_size,
    ActivationType act_type //
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(batch_idx >= batch_size || neuron_idx >= neuron_size)
        return;

    int idx = batch_idx * neuron_size + neuron_idx;
    activated_g[idx] *= activationDer(pre_activated[idx], act_type);
}

__global__ void bucketed_bwd_biases_kernel( //
    const float *activated_g,               // [batch_size, neuron_size]
    float *biases_g,                        // [neuron_size, bucket_size]
    const int *bucket_idxs,                 // [batch_size]
    int batch_size,
    int neuron_size,
    int bucket_size //
) {
    extern __shared__ float bias_shared[];

    int bucket_idx = blockIdx.x;
    int neuron_idx = blockIdx.y;
    int tid = threadIdx.x;

    if(bucket_idx >= bucket_size || neuron_idx >= neuron_size)
        return;

    float local_sum = 0.0f;

    // sum gradients for this neuron and bucket across all batch samples
    for(int b = tid; b < batch_size; b += blockDim.x)
        if(bucket_idxs[b] == bucket_idx)
            local_sum += activated_g[b * neuron_size + neuron_idx];

    bias_shared[tid] = local_sum;
    __syncthreads();

    // reduce within block
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(tid < stride)
            bias_shared[tid] += bias_shared[tid + stride];
        __syncthreads();
    }

    if(tid == 0)
        biases_g[neuron_idx * bucket_size + bucket_idx] = bias_shared[0];
}

__global__ void bucketed_bwd_input_kernel( //
    const float *activated_g,              // [batch_size, neuron_size]
    const float *weights_v,                // [bucket_size, neuron_size, input_size]
    float *input_g,                        // [batch_size, input_size]
    const int *bucket_idxs,                // [batch_size]
    int batch_size,
    int input_size,
    int neuron_size //
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int input_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(batch_idx >= batch_size || input_idx >= input_size)
        return;

    const int bucket_idx = bucket_idxs[batch_idx];
    const int bucket_offset = bucket_idx * neuron_size * input_size;

    // compute gradient w.r.t. input
    float sum = 0.0f;
    for(int i = 0; i < neuron_size; ++i) {
        int output_offset = batch_idx * neuron_size + i;
        int weight_offset = bucket_offset + i * input_size + input_idx;
        sum += activated_g[output_offset] * weights_v[weight_offset];
    }

    input_g[batch_idx * input_size + input_idx] = sum;
}

__global__ void bucketed_bwd_weights_kernel( //
    const float *activated_g,                // [batch_size, neuron_size]
    const float *input_v,                    // [batch_size, input_size]
    float *weights_g,                        // [bucket_size, neuron_size, input_size]
    const int *bucket_idxs,                  // [batch_size]
    int batch_size,
    int input_size,
    int neuron_size,
    int bucket_size //
) {
    int bucket_idx = blockIdx.x;
    int neuron_idx = blockIdx.y;
    int input_idx = blockIdx.z;

    if(bucket_idx >= bucket_size || neuron_idx >= neuron_size || input_idx >= input_size)
        return;

    extern __shared__ float shared_grad[];

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    float local_sum = 0.0f;

    // each thread processes multiple batch samples
    for(int b = tid; b < batch_size; b += block_size) {
        if(bucket_idxs[b] == bucket_idx) {
            local_sum += activated_g[b * neuron_size + neuron_idx] * input_v[b * input_size + input_idx];
        }
    }

    shared_grad[tid] = local_sum;
    __syncthreads();

    // reduction within block
    for(int stride = block_size / 2; stride > 0; stride >>= 1) {
        if(tid < stride)
            shared_grad[tid] += shared_grad[tid + stride];
        __syncthreads();
    }

    if(tid == 0) {
        int weight_idx = bucket_idx * neuron_size * input_size + neuron_idx * input_size + input_idx;
        weights_g[weight_idx] = shared_grad[0];
    }
}

void bucketed_affine_bwd( //
    Tensor &weights,
    Tensor &biases,
    Tensor &inputs,
    Tensor &activated,
    DenseMatrix &pre_activated,
    const Array<int> &bucket_indices,
    const ActivationType act_type,
    const int bucket_size //
) {
    DenseMatrix &weights_v = weights.getValues();
    DenseMatrix &weights_g = weights.getGradients();

    DenseMatrix &biases_g = biases.getGradients();

    DenseMatrix &inputs_v = inputs.getValues();
    DenseMatrix &inputs_g = inputs.getGradients();

    DenseMatrix &activated_v = activated.getValues();
    DenseMatrix &activated_g = activated.getGradients();

    const int batch_size = inputs_v.numRows();
    const int input_size = inputs_v.numCols();
    const int neuron_size = weights_v.numRows() / bucket_size;

    ASSERT(activated_g.numCols() == neuron_size && biases_g.numCols() == 1);

    ASSERT(weights_g.numCols() == input_size &&    //
           neuron_size == activated_g.numCols() && //
           batch_size == activated_g.numRows());

    ASSERT(weights_v.devAddress() &&   //
           weights_g.devAddress() &&   //
           biases_g.devAddress() &&    //
           inputs_v.devAddress() &&    //
           inputs_g.devAddress() &&    //
           activated_v.devAddress() && //
           activated_g.devAddress() && //
           pre_activated.devAddress());

    // step 1: apply activation derivative to get gradients w.r.t. pre-activation
    if(act_type != Linear) {
        dim3 block_size(16, 16);
        dim3 grid_size((batch_size + block_size.x - 1) / block_size.x, //
                       (neuron_size + block_size.y - 1) / block_size.y);

        apply_activation_derivative_kernel<<<grid_size, block_size>>>( //
            activated_g.devAddress(),
            pre_activated.devAddress(),
            batch_size,
            neuron_size,
            act_type);
    }

    // step 2: compute bias gradients using grad_pre_activated
    {
        const int block_size = 256;
        dim3 grid_size(bucket_size, neuron_size);

        int shared_mem_size = block_size * sizeof(float);

        bucketed_bwd_biases_kernel<<<grid_size, block_size, shared_mem_size>>>( //
            activated_g.devAddress(),
            biases_g.devAddress(),
            bucket_indices.devAddress(),
            batch_size,
            neuron_size,
            bucket_size);
    }

    // step 3: compute weight gradients using grad_pre_activated
    {
        const int block_size = 256;
        dim3 grid_size(bucket_size, neuron_size, input_size);

        int shared_mem_size = block_size * sizeof(float);

        bucketed_bwd_weights_kernel<<<grid_size, block_size, shared_mem_size>>>( //
            activated_g.devAddress(),
            inputs_v.devAddress(),
            weights_g.devAddress(),
            bucket_indices.devAddress(),
            batch_size,
            input_size,
            neuron_size,
            bucket_size);
    }

    // step 4: compute input gradients using grad_pre_activated
    {
        dim3 block_size(16, 16);
        dim3 grid_size((batch_size + block_size.x - 1) / block_size.x, //
                       (input_size + block_size.y - 1) / block_size.y);

        bucketed_bwd_input_kernel<<<grid_size, block_size>>>( //
            activated_g.devAddress(),
            weights_v.devAddress(),
            inputs_g.devAddress(),
            bucket_indices.devAddress(),
            batch_size,
            input_size,
            neuron_size);
    }

    cudaDeviceSynchronize();
}
