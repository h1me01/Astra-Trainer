#include <cstdlib>

#include "cublas.h"

#define CUBLAS_CHECK(expr)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = (expr);                                                                                \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            printf("CUBLAS error calling %s\n", #expr);                                                                \
            printf("    file:  %s\n", __FILE__);                                                                       \
            printf("    line:  %d\n", __LINE__);                                                                       \
            printf("    error: %s\n", cublasGetStatusString(status));                                                  \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

namespace kernel::cublas {

namespace {
cublasHandle_t handle;
}

void create() {
    CUBLAS_CHECK(cublasCreate(&handle));
}

void destroy() {
    CUBLAS_CHECK(cublasDestroy(handle));
}

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
) {
    CUBLAS_CHECK(cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

} // namespace kernel::cublas
