#pragma once

#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "../util.h"

namespace kernel::cublas {

void create();
void destroy();

void sgemm(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float beta,
    float* C,
    int ldc
);

} // namespace kernel::cublas
