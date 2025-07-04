#pragma once

#include "../../nn/data/include.h"

void select_fwd( //
    const DenseMatrix<float> &input_v,
    DenseMatrix<float> &output_v,
    const Array<int> &indices,
    const int batch_size,
    const int input_size,
    const int output_size);

void select_bwd( //
    DenseMatrix<float> &input_g,
    const DenseMatrix<float> &output_g,
    const Array<int> &indices,
    const int batch_size,
    const int input_size,
    const int output_size);
