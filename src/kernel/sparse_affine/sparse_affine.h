#pragma once

#include "../../data/include.h"
#include "../elemwise/unary/unary.h"
#include "../util.h"

namespace kernel {

void sparse_affine_fwd(
    const DenseMatrix& weights_d,
    const DenseMatrix& biases_d,
    DenseMatrix& out_d,
    const SparseMatrix& indices,
    const int out_offset,
    ActOp op
);

void sparse_affine_bwd(
    DenseMatrix& weights_g,
    DenseMatrix& biases_g,
    const Tensor& out,
    const SparseMatrix& indices,
    const int out_offset,
    ActOp op
);

} // namespace kernel