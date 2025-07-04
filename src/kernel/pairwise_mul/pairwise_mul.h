#pragma once

#include "../../nn/data/include.h"

void pairwise_mul_fwd( //
    const DenseMatrix<float> &inputs_v,
    DenseMatrix<float> &output_v);

void pairwise_mul_bwd( //
    Tensor &inputs,
    const DenseMatrix<float> output_g);
