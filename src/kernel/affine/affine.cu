#include "affine.h"

// cublasSgemm performs C = A * B * alpha + C * beta

cublasHandle_t CUBLAS_HANDLE;

void create_cublas() {
    cublasCreate(&CUBLAS_HANDLE);
}

void destroy_cublas() {
    cublasDestroy(CUBLAS_HANDLE);
}
