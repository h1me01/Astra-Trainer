#include "loss.h"

constexpr int block_size = 1024;

// MPE

__global__ void mpe_kernel(        //
    const float *targets,          //
    const float *output_v,         //
    float *output_g,               //
    float *loss,                   //
    const float power,             //
    const ActivationType act_type, //
    const int size                 //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    const float pre_activated = output_v[idx];
    const float activated = activate(pre_activated, act_type);
    const float diff = activated - targets[idx];
    const float abs_diff = abs(diff);

    float grad = powf(abs_diff, power - 1.0f) * power * activate_der(pre_activated, act_type);
    output_g[idx] = (diff > 0 ? 1 : -1) * grad;

    atomicAdd(loss, powf(abs_diff, power));
}

void mpe_loss(                     //
    const Array<float> &targets,   //
    Array<float> &loss,            //
    Tensor &output,                //
    const float power,             //
    const ActivationType act_type, //
    const int size                 //
) {
    const DenseMatrix<float> &output_v = output.get_data();
    DenseMatrix<float> &output_g = output.get_grads();

    ASSERT(output_v.dev_address()    //
           && output_g.dev_address() //
           && targets.dev_address()  //
           && loss.dev_address());

    const int grid_size = std::ceil((float) output_v.size() / block_size);

    mpe_kernel<<<grid_size, block_size>>>( //
        targets.dev_address(),
        output_v.dev_address(),
        output_g.dev_address(),
        loss.dev_address(),
        power,
        act_type,
        output_v.size());
}

// MSE

__global__ void mse_kernel(        //
    const float *targets,          //
    const float *output_v,         //
    float *output_g,               //
    float *loss,                   //
    const ActivationType act_type, //
    const int size                 //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    const float pre_activated = output_v[idx];
    const float activated = activate(pre_activated, act_type);
    const float diff = activated - targets[idx];

    output_g[idx] = 2.0f * diff * activate_der(pre_activated, act_type);

    atomicAdd(loss, diff * diff);
}

void mse_loss(                     //
    const Array<float> &targets,   //
    Array<float> &loss,            //
    Tensor &output,                //
    const ActivationType act_type, //
    const int size                 //
) {
    const DenseMatrix<float> &output_v = output.get_data();
    DenseMatrix<float> &output_g = output.get_grads();

    ASSERT(output_v.dev_address()    //
           && output_g.dev_address() //
           && targets.dev_address()  //
           && loss.dev_address());

    const int grid_size = std::ceil((float) output_v.size() / block_size);

    mse_kernel<<<grid_size, block_size>>>( //
        targets.dev_address(),
        output_v.dev_address(),
        output_g.dev_address(),
        loss.dev_address(),
        act_type,
        output_v.size());
}
