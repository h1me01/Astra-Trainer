#pragma once

#include "../../data/include.h"
#include "../activation/activation.h"
#include "../util.h"

namespace kernel {

// fairly specific fusion for typical multi-layer models

void sparse_affine_pairwise_mul_fwd(
    const DenseMatrix& weights_d,
    const DenseMatrix& biases_d,
    DenseMatrix& out_d,
    const SparseMatrix& indices,
    const int out_offset,
    const ActivationType act_type
);

void sparse_affine_pairwise_mul_bwd(
    const DenseMatrix& weights_d,
    DenseMatrix& weights_g,
    Tensor& biases,
    const DenseMatrix& out_g,
    const SparseMatrix& indices,
    const int out_offset,
    const ActivationType act_type
);

} // namespace kernel
