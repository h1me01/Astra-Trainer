#include "bucketed_affine.h"

// AFFINE

__global__ void bucketed_affine_kernel( //
    const float *input_v,               // [batch_size, input_size]
    const float *weights_v,             // [bucket_size, neuron_size, input_size]
    const float *biases_v,              // [neuron_size, 1]
    float *activated_v,                 // [batch_size, neuron_size]
    float *pre_activated,               // [batch_size, neuron_size]
    const int *bucket_idxs,             // [batch_size]
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

    const int bucket_idx = bucket_idxs[batch_idx];
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
    // clang-format on
    const int batch_size = inputs_v.numCols();
    const int input_size = inputs_v.numRows();
    const int neuron_size = weights_v.numRows() / bucket_size;

    ASSERT(activated_v.numRows() == neuron_size && biases_v.numCols() == 1);

    ASSERT(weights_v.numCols() == input_size &&    //
           neuron_size == activated_v.numRows() && //
           batch_size == activated_v.numCols());

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

    // cudaDeviceSynchronize();
}

// AFFINE BW

__global__ void bucketed_bwd_input_kernel( //
    const float *grad_output,              // [batch_size, neuron_size]
    const float *weights,                  // [bucket_size, neuron_size, input_size]
    float *grad_input,                     // [batch_size, input_size]
    const int *bucket_idxs,                // [batch_size]
    int batch_size,
    int input_size,
    int neuron_size //
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int input_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(batch_idx >= batch_size || input_idx >= input_size)
        return;

    const int bucket_offset = bucket_idxs[batch_idx] * neuron_size * input_size;

    // compute gradient w.r.t. input
    float sum = 0.0f;
    for(int i = 0; i < neuron_size; ++i) {
        int grad_output_offset = batch_idx * neuron_size + i;
        int weight_offset = bucket_offset + i * input_size + input_idx;
        sum += grad_output[grad_output_offset] * weights[weight_offset];
    }

    grad_input[batch_idx * input_size + input_idx] = sum;
}

__global__ void bucketed_bwd_weights_kernel( //
    const float *grad_pre_activated,         // [batch_size, neuron_size] (after activation derivative)
    const float *input,                      // [batch_size, input_size]
    float *grad_weights,                     // [bucket_size, neuron_size, input_size]
    const int *bucket_idxs,                  // [batch_size]
    int batch_size,
    int input_size,
    int neuron_size,
    int bucket_size //
) {
    extern __shared__ float shared_data[];

    int bucket_idx = blockIdx.x;
    int neuron_idx = blockIdx.y;
    int input_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if(bucket_idx >= bucket_size || neuron_idx >= neuron_size || input_idx >= input_size)
        return;

    float local_sum = 0.0f;
    int tid = threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;

    // each thread processes multiple batch samples with stride
    for(int b = threadIdx.x * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.x * blockDim.y) {
        if(bucket_idxs[b] == bucket_idx) {
            local_sum += grad_pre_activated[b * neuron_size + neuron_idx] * input[b * input_size + input_idx];
        }
    }

    shared_data[tid] = local_sum;
    __syncthreads();

    // reduction within block - only reduce across batch dimension
    int reduction_size = blockDim.x * blockDim.y;
    for(int stride = reduction_size / 2; stride > 0; stride >>= 1) {
        if(tid < stride * blockDim.z && tid + stride * blockDim.z < blockDim.x * blockDim.y * blockDim.z) {
            shared_data[tid] += shared_data[tid + stride * blockDim.z];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0 && threadIdx.y == 0) {
        grad_weights[bucket_idx * neuron_size * input_size + neuron_idx * input_size + input_idx] =
            shared_data[threadIdx.z];
    }
}

__global__ void apply_activation_derivative_kernel( //
    const float *grad_output,                       // [batch_size, neuron_size]
    const float *pre_activated,                     // [batch_size, neuron_size]
    float *grad_pre_activated,                      // [batch_size, neuron_size]
    int batch_size,
    int neuron_size,
    ActivationType act_type //
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(batch_idx >= batch_size || neuron_idx >= neuron_size)
        return;

    int idx = batch_idx * neuron_size + neuron_idx;
    float pre_act_val = pre_activated[idx];
    float grad_out = grad_output[idx];

    // Apply chain rule: grad_pre_activated = grad_output * activation_derivative(pre_activated)
    grad_pre_activated[idx] = grad_out * activationDer(pre_act_val, act_type);
}

// Separate kernel for bias gradients (using gradients w.r.t. pre-activation)
__global__ void bucketed_bwd_biases_kernel( //
    const float *grad_pre_activated,        // [batch_size, neuron_size]
    float *grad_biases,                     // [neuron_size, bucket_size]
    const int *bucket_idxs,                 // [batch_size]
    int batch_size,
    int neuron_size,
    int bucket_size //
) {
    extern __shared__ float bias_shared[];

    int bucket_idx = blockIdx.x;
    int neuron_idx = blockIdx.y;

    if(bucket_idx >= bucket_size || neuron_idx >= neuron_size)
        return;

    int tid = threadIdx.x;
    float local_sum = 0.0f;

    // Sum gradients for this neuron and bucket across all batch samples
    for(int b = tid; b < batch_size; b += blockDim.x) {
        if(bucket_idxs[b] == bucket_idx) {
            local_sum += grad_pre_activated[b * neuron_size + neuron_idx];
        }
    }

    bias_shared[tid] = local_sum;
    __syncthreads();

    // Reduce within block
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(tid < stride) {
            bias_shared[tid] += bias_shared[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0) {
        grad_biases[neuron_idx * bucket_size + bucket_idx] = bias_shared[0];
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
    // clang-format on
    const DenseMatrix &weights_v = weights.getValues();
    DenseMatrix &weights_g = weights.getGradients();

    DenseMatrix &biases_g = biases.getGradients();

    const DenseMatrix &inputs_v = inputs.getValues();
    DenseMatrix &inputs_g = inputs.getGradients();

    const DenseMatrix &activated_v = activated.getValues();
    const DenseMatrix &activated_g = activated.getGradients();

    const int batch_size = inputs_v.numCols();
    const int input_size = inputs_v.numRows();
    const int neuron_size = weights_v.numRows() / bucket_size;

    ASSERT(activated_g.numRows() == neuron_size && biases_g.numCols() == 1);

    ASSERT(weights_g.numCols() == inputs_g.numRows() && //
           neuron_size == activated_g.numRows() &&      //
           inputs_g.numCols() == activated_g.numCols());

    ASSERT(weights_v.devAddress() &&   //
           weights_g.devAddress() &&   //
           biases_g.devAddress() &&    //
           inputs_v.devAddress() &&    //
           inputs_g.devAddress() &&    //
           activated_v.devAddress() && //
           activated_g.devAddress() && //
           pre_activated.devAddress());

    // Allocate temporary storage for gradients w.r.t. pre-activation
    DenseMatrix grad_pre_activated(neuron_size, batch_size);

    // Step 1: Apply activation derivative to get gradients w.r.t. pre-activation
    {
        dim3 block_size(16, 16);
        dim3 grid_size((batch_size + block_size.x - 1) / block_size.x, //
                       (neuron_size + block_size.y - 1) / block_size.y);

        apply_activation_derivative_kernel<<<grid_size, block_size>>>( //
            activated_g.devAddress(),
            pre_activated.devAddress(),
            grad_pre_activated.devAddress(),
            batch_size,
            neuron_size,
            act_type);
    }

    // Step 2: Compute bias gradients using grad_pre_activated
    {
        dim3 block_size(256); // 1D block for bias computation
        dim3 grid_size(bucket_size, neuron_size);

        int shared_mem_size = block_size.x * sizeof(float);

        bucketed_bwd_biases_kernel<<<grid_size, block_size, shared_mem_size>>>( //
            grad_pre_activated.devAddress(),
            biases_g.devAddress(),
            bucket_indices.devAddress(),
            batch_size,
            neuron_size,
            bucket_size);
    }

    // Step 3: Compute weight gradients using grad_pre_activated
    {
        dim3 block_size(4, 4, 4);
        dim3 grid_size((bucket_size + block_size.x - 1) / block_size.x, //
                       (neuron_size + block_size.y - 1) / block_size.y,
                       (input_size + block_size.z - 1) / block_size.z);

        int shared_mem_size = block_size.x * block_size.y * block_size.z * sizeof(float);

        bucketed_bwd_weights_kernel<<<grid_size, block_size, shared_mem_size>>>( //
            grad_pre_activated.devAddress(),
            inputs_v.devAddress(),
            weights_g.devAddress(),
            bucket_indices.devAddress(),
            batch_size,
            input_size,
            neuron_size,
            bucket_size);
    }

    // Step 4: Compute input gradients using grad_pre_activated
    {
        dim3 block_size(16, 16);
        dim3 grid_size((batch_size + block_size.x - 1) / block_size.x, //
                       (input_size + block_size.y - 1) / block_size.y);

        bucketed_bwd_input_kernel<<<grid_size, block_size>>>( //
            grad_pre_activated.devAddress(),
            weights_v.devAddress(),
            inputs_g.devAddress(),
            bucket_indices.devAddress(),
            batch_size,
            input_size,
            neuron_size);
    }

    cudaDeviceSynchronize();
}