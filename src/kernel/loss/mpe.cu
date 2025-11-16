#include "loss.h"

namespace kernel {

constexpr int block_size = 1024;

__global__ void mpe_kernel(        //
    const float *targets,          //
    const float *out_v,            //
    float *out_g,                  //
    float *loss,                   //
    const float power,             //
    const ActivationType act_type, //
    const int size                 //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    const float pred = out_v[idx];
    const float act = activate(pred, act_type);
    const float diff = act - targets[idx];
    const float abs_diff = fabsf(diff);

    const float grad_magnitude = power * powf(abs_diff, power - 1.0f);
    const float sign_diff = (diff > 0.0f) ? 1.0f : -1.0f;
    out_g[idx] = grad_magnitude * sign_diff * activate_der(pred, act_type);

    float p = powf(abs_diff, power);
    if(p != 0.0f)
        atomicAdd(loss, p);
}

void mpe_loss(                    //
    const Array<float> &targets,  //
    Array<float> &loss,           //
    Tensor<float> &out,           //
    const float power,            //
    const ActivationType act_type //
) {
    const auto &out_v = out.get_values();
    auto &out_g = out.get_gradients();

    ASSERT(out_v.is_dev_allocated() &&   //
           out_g.is_dev_allocated() &&   //
           targets.is_dev_allocated() && //
           loss.is_dev_allocated());

    const int blocks = get_num_blocks(out_v.size(), block_size);
    mpe_kernel<<<blocks, block_size>>>( //
        targets.dev_address(),
        out_v.dev_address(),
        out_g.dev_address(),
        loss.dev_address(),
        power,
        act_type,
        out_v.size());
}

} // namespace kernel
