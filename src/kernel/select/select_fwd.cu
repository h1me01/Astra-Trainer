#include "select.h"

constexpr int block_size = 1024;

__global__ void select_fwd_kernel( //
    const float *inputs_v,         //
    float *output_v,               //
    const int *indices,            //
    const int batch_size,          //
    const int input_size,          //
    const int output_size          //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch_size * output_size)
        return;

    const int batch_idx = idx / output_size;
    const int output_idx = idx % output_size;

    const int bucket = indices[batch_idx];
    const int input_offset = input_size * batch_idx + output_size * bucket + output_idx;

    output_v[output_size * batch_idx + output_idx] = inputs_v[input_offset];
}

void select_fwd(                        //
    const DenseMatrix<float> &inputs_v, //
    DenseMatrix<float> &output_v,       //
    const Array<int> &indices           //
) {
    ASSERT(inputs_v.dev_address() && //
           output_v.dev_address() && //
           indices.dev_address());

    const int batch_size = output_v.cols();
    const int output_size = output_v.rows();

    const int grid_size = std::ceil(float(batch_size * output_size) / block_size);

    select_fwd_kernel<<<grid_size, block_size>>>( //
        inputs_v.dev_address(),
        output_v.dev_address(),
        indices.dev_address(),
        batch_size,
        inputs_v.rows(),
        output_size);
}
