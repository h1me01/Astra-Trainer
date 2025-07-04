#pragma once

#include "../../nn/data/include.h"

void select_fwd( //
    const DenseMatrix<float> &inputs_v,
    DenseMatrix<float> &output_v,
    const Array<int> &indices);

void select_bwd( //
    DenseMatrix<float> &inputs_g,
    const DenseMatrix<float> &output_g,
    const Array<int> &indices);
