#pragma once

#include "../../data/include.h"
#include "../cublas/cublas.h"
#include "../util.h"

namespace kernel {

void affine_fwd(DenseMatrix& weights_d, DenseMatrix& biases_d, const DenseMatrix& inputs_d, DenseMatrix& out_d);
void affine_bwd(Tensor& weights, Tensor& biases, Tensor& in, Tensor& out);

} // namespace kernel
