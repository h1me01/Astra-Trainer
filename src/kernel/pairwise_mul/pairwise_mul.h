#pragma once

#include "../../nn/data/include.h"

void pairwise_mul_fwd_kernel( //
    const DenseMatrix<float> &inputs_v,
    DenseMatrix<float> &output_v);

void pairwise_mul_bwd_kernel( //
    Tensor &inputs,
    const DenseMatrix<float> output_g);
