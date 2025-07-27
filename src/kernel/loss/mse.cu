#include "loss.h"

constexpr int block_size = 1024;

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

void mse_loss(                    //
    const Array<float> &targets,  //
    Array<float> &loss,           //
    Tensor<float> &output,        //
    const ActivationType act_type //
) {
    const auto &output_v = output.get_data();
    auto &output_g = output.get_grads();

    ASSERT(output_v.dev_address() && //
           output_g.dev_address() && //
           targets.dev_address() &&  //
           loss.dev_address());

    const int grid_size = std::ceil((float) output_v.size() / block_size);

    mse_kernel<<<grid_size, block_size>>>( //
        targets.dev_address(),
        output_v.dev_address(),
        output_g.dev_address(),
        loss.dev_address(),
        act_type,
        output_v.size());
}
