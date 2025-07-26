#include "../util.h"
#include "optimizer.h"

float get_decay(const float lr, const float decay) {
    return 1.0f - lr * decay;
}

constexpr int block_size = 1024;

// ADAM

__global__ void adam_kernel( //
    float *vals,             //
    const float *grads,      //
    float *moms,             //
    float *vels,             //
    const float lr,          //
    const float beta1,       //
    const float beta2,       //
    const float eps,         //
    const float decay,       //
    const float min_val,     //
    const float max_val,     //
    const float grad_scale,  //
    const int size           //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    const float grad = grads[idx] * grad_scale;

    float val = vals[idx] * decay;

    float mom = moms[idx];
    float vel = vels[idx];

    mom = beta1 * mom + (1.0f - beta1) * grad;
    vel = beta2 * vel + (1.0f - beta2) * grad * grad;

    val -= lr * mom / (sqrtf(vel) + eps);

    moms[idx] = mom;
    vels[idx] = vel;
    vals[idx] = clamp(val, min_val, max_val);
}

void adam_optim(               //
    DenseMatrix<float> &vals,  //
    DenseMatrix<float> &grads, //
    Array<float> &moms,        //
    Array<float> &vels,        //
    const OptimParams &params, //
    const float min_val,       //
    const float max_val,       //
    const float grad_scale     //
) {
    ASSERT(vals.dev_address() &&  //
           grads.dev_address() && //
           moms.dev_address() &&  //
           vels.dev_address());

    const int grid_size(std::ceil((float) vals.size() / block_size));

    adam_kernel<<<grid_size, block_size>>>( //
        vals.dev_address(),
        grads.dev_address(),
        moms.dev_address(),
        vels.dev_address(),
        params.lr,
        params.beta1,
        params.beta2,
        params.eps,
        get_decay(params.lr, params.decay),
        min_val,
        max_val,
        grad_scale,
        vals.size());

    grads.clear_dev();
}
