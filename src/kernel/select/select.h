#pragma once

#include "../../nn/data.h"

void select_fwd( //
    const DenseMatrix &input_v,
    DenseMatrix &output_v,
    const Array<int> &bucket_indices,
    const int batch_size,
    const int input_size,
    const int output_size);

void select_bwd( //
    DenseMatrix &input_g,
    const DenseMatrix &output_g,
    const Array<int> &bucket_indices,
    const int batch_size,
    const int input_size,
    const int output_size);
