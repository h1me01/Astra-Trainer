#include "../util.h"
#include "optimizer.h"

float getDecay(const float lr, const float decay) {
    return 1.0f - lr * decay;
}

const int block_size = 1024;

// ADAM

__global__ void adam_kernel( //
    float *vals,             //
    float *grads,            //
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

    float val = vals[idx];
    val *= decay;

    if(grad == 0.0f)
        return;

    float mom = moms[idx];
    float vel = vels[idx];

    mom = beta1 * mom + (1.0f - beta1) * grad;
    vel = beta2 * vel + (1.0f - beta2) * grad * grad;

    val -= lr * mom / (sqrtf(vel) + eps);

    moms[idx] = mom;
    vels[idx] = vel;
    vals[idx] = clamp(val, min_val, max_val);
    grads[idx] = 0.0f;
}

void adam_optim(               //
    DenseMatrix<float> &vals,  //
    DenseMatrix<float> &grads, //
    Array<float> &moms,        //
    Array<float> &vels,        //
    const AdamParams &params,  //
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
        getDecay(params.lr, params.decay),
        min_val,
        max_val,
        grad_scale,
        vals.size());
}

// RADAM

// https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py
__global__ void radam_kernel(  //
    float *vals,               //
    float *grads,              //
    float *moms,               //
    float *vels,               //
    const float lr,            //
    const float beta1,         //
    const float beta2,         //
    const float eps,           //
    const float decay,         //
    const float min_val,       //
    const float max_val,       //
    const float grad_scale,    //
    const int N_sma_threshold, //
    const int step,            //
    const int size             //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    const float grad = grads[idx] * grad_scale;

    float val = vals[idx];
    val *= decay;

    if(grad == 0.0f)
        return;

    float mom = moms[idx];
    float vel = vels[idx];

    mom = beta1 * mom + (1.0f - beta1) * grad;
    vel = beta2 * vel + (1.0f - beta2) * grad * grad;

    float beta2_t = powf(beta2, step);
    float N_sma_max = 2.0f / (1.0f - beta2) - 1.0f;
    float N_sma = N_sma_max - 2.0f * step * beta2_t / (1.0f - beta2_t);

    if(N_sma >= N_sma_threshold) {
        // clang-format off
        float step_size = lr 
                        * sqrtf((1.0f - beta2_t) * (N_sma - 4.0f) / (N_sma_max - 4.0f) 
                        * (N_sma - 2.0f) / N_sma * N_sma_max / (N_sma_max - 2.0f)) 
                        / (1.0f - powf(beta1, step));
        // clang-format on
        float denom = sqrtf(vel) + eps;
        val -= step_size * mom / denom;
    } else {
        float step_size = lr * (1.0f - powf(beta1, step));
        val -= step_size * mom;
    }

    moms[idx] = mom;
    vels[idx] = vel;
    vals[idx] = clamp(val, min_val, max_val);
    grads[idx] = 0.0f;
}

void radam_optim(              //
    DenseMatrix<float> &vals,  //
    DenseMatrix<float> &grads, //
    Array<float> &moms,        //
    Array<float> &vels,        //
    const AdamParams &params,  //
    const float min_val,       //
    const float max_val,       //
    const float grad_scale,    //
    const int N_sma_threshold, //
    const int step             //
) {
    ASSERT(vals.dev_address() &&  //
           grads.dev_address() && //
           moms.dev_address() &&  //
           vels.dev_address());

    const int grid_size(std::ceil((float) vals.size() / block_size));

    radam_kernel<<<grid_size, block_size>>>( //
        vals.dev_address(),
        grads.dev_address(),
        moms.dev_address(),
        vels.dev_address(),
        params.lr,
        params.beta1,
        params.beta2,
        params.eps,
        getDecay(params.lr, params.decay),
        min_val,
        max_val,
        grad_scale,
        N_sma_threshold,
        step,
        vals.size());
}

// RANGER

// https://github.com/official-stockfish/nnue-pytorch/blob/master/ranger.py
__global__ void ranger_kernel( //
    float *vals,               //
    float *grads,              //
    float *moms,               //
    float *vels,               //
    float *slow_buffer,        //
    const float lr,            //
    const float beta1,         //
    const float beta2,         //
    const float eps,           //
    const float decay,         //
    const float min_val,       //
    const float max_val,       //
    const float grad_scale,    //
    const float alpha,         //
    const int k,               //
    const int N_sma_threshold, //
    const int step,            //
    const int size             //
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;

    const float grad = grads[idx] * grad_scale;
    float val = vals[idx];
    val *= decay;

    if(grad == 0.0f)
        return;

    float mom = moms[idx];
    float vel = vels[idx];

    mom = beta2 * mom + (1.0 - beta2) * grad * grad;
    vel = beta1 * vel + (1.0 - beta1) * grad;

    float beta2_t = powf(beta2, step);
    float beta1_correction = 1.0f - powf(beta1, step);
    float N_sma_max = 2.0f / (1.0f - beta2) - 1.0f;
    float N_sma = N_sma_max - 2.0f * step * beta2_t / (1.0f - beta2_t);

    if(N_sma >= N_sma_threshold) {
        // clang-format off
        float step_size = lr 
                        * sqrtf((1.0f - beta2_t) * (N_sma - 4.0f) / (N_sma_max - 4.0f) 
                        * (N_sma - 2.0f) / N_sma * N_sma_max / (N_sma_max - 2.0f)) 
                        / beta1_correction;
        // clang-format on
        val -= step_size * mom / (sqrtf(vel) + eps);
    } else {
        val -= (lr / beta1_correction) * mom;
    }

    // moving average of weights
    if(step % k == 0) {
        float slow = slow_buffer[idx];
        slow += alpha * (val - slow);
        slow_buffer[idx] = slow;
        val = slow;
    }

    moms[idx] = mom;
    vels[idx] = vel;
    vals[idx] = clamp(val, min_val, max_val);
    grads[idx] = 0.0f;
}

void ranger_optim(             //
    DenseMatrix<float> &vals,  //
    DenseMatrix<float> &grads, //
    Array<float> &moms,        //
    Array<float> &vels,        //
    Array<float> &slow_buffer, //
    const AdamParams &params,  //
    const float min_val,       //
    const float max_val,       //
    const float grad_scale,    //
    const float alpha,         //
    const int k,               //
    const int N_sma_threshold, //
    const int step             //
) {
    ASSERT(vals.dev_address() &&  //
           grads.dev_address() && //
           moms.dev_address() &&  //
           vels.dev_address() &&  //
           slow_buffer.dev_address());

    const int grid_size(std::ceil((float) vals.size() / block_size));

    ranger_kernel<<<grid_size, block_size>>>( //
        vals.dev_address(),
        grads.dev_address(),
        moms.dev_address(),
        vels.dev_address(),
        slow_buffer.dev_address(),
        params.lr,
        params.beta1,
        params.beta2,
        params.eps,
        getDecay(params.lr, params.decay),
        min_val,
        max_val,
        grad_scale,
        alpha,
        k,
        N_sma_threshold,
        step,
        vals.size());
}
