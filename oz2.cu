#include "gpu/mxGPUArray.h"
#include "include/gemmul8.hpp"
#include "mex.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    //=====
    // input
    //=====
    mxInitGPU();
    mxGPUArray const *A   = mxGPUCreateFromMxArray(prhs[0]); // create mxGPUArray object from A
    mxGPUArray const *B   = mxGPUCreateFromMxArray(prhs[1]); // create mxGPUArray object from B
    double num_moduli_d   = (double)mxGetScalar(prhs[2]);
    double fastmode_d     = (double)mxGetScalar(prhs[3]);
    mwSize const ndims    = mxGPUGetNumberOfDimensions(A); // #dimensions of A
    mxClassID const cid   = mxGPUGetClassID(A);            // specifying the element class
    mxComplexity const cx = mxGPUGetComplexity(A);         // specifying the complexity

    const mwSize *dimA = mxGPUGetDimensions(A); // dimensions of A
    const mwSize *dimB = mxGPUGetDimensions(B); // dimensions of B
    const size_t m     = dimA[0];               // #rows of A
    const size_t k     = dimA[1];               // #columns of A
    const size_t n     = dimB[1];               // #columns of B

    size_t dimC[2]             = {m, n};                                          // dimensions of C
    mxGPUInitialize const init = MX_GPU_DO_NOT_INITIALIZE;                        // specifying whether to initialize elements values to 0
    mxGPUArray *C              = mxGPUCreateGPUArray(ndims, dimC, cid, cx, init); // create mxGPUArray object

    cublasHandle_t ch;
    cublasCreate(&ch);

    const unsigned num_moduli = (const unsigned)num_moduli_d;
    const bool fastmode       = (fastmode_d == 1.0) ? true : false;
    size_t worksize           = gemmul8::workSize<false>(m, n, k, num_moduli, false, false, nullptr, nullptr);
    void *work                = nullptr;
    cudaMalloc(&work, worksize);

    if (cid == mxDOUBLE_CLASS) {

        using TYPE                   = double;
        const cublasOperation_t op_A = CUBLAS_OP_N;
        const cublasOperation_t op_B = CUBLAS_OP_N;
        const TYPE alpha             = (TYPE)1;
        const TYPE *alpha_ptr        = &alpha;
        const TYPE beta              = (TYPE)0;
        const TYPE *beta_ptr         = &beta;
        const TYPE *devA             = (const TYPE *)mxGPUGetDataReadOnly(A); // read-only pointer to the underlying data of A
        const TYPE *devB             = (const TYPE *)mxGPUGetDataReadOnly(B); // read-only pointer to the underlying data of B
        TYPE *devC                   = (TYPE *)mxGPUGetData(C);               // pointer to the underlying data of devC

        cudaDeviceSynchronize();
        gemmul8::gemm<TYPE>(ch,
                            op_A, op_B, m, n, k,
                            alpha_ptr,
                            devA, m,
                            devB, k,
                            beta_ptr,
                            devC, m,
                            num_moduli, fastmode, work);

    } else if (cid == mxSINGLE_CLASS) {

        using TYPE                   = float;
        const cublasOperation_t op_A = CUBLAS_OP_N;
        const cublasOperation_t op_B = CUBLAS_OP_N;
        const TYPE alpha             = (TYPE)1;
        const TYPE *alpha_ptr        = &alpha;
        const TYPE beta              = (TYPE)0;
        const TYPE *beta_ptr         = &beta;
        const TYPE *devA             = (const TYPE *)mxGPUGetDataReadOnly(A); // read-only pointer to the underlying data of A
        const TYPE *devB             = (const TYPE *)mxGPUGetDataReadOnly(B); // read-only pointer to the underlying data of B
        TYPE *devC                   = (TYPE *)mxGPUGetData(C);               // pointer to the underlying data of devC

        cudaDeviceSynchronize();
        gemmul8::gemm<TYPE>(ch,
                            op_A, op_B, m, n, k,
                            alpha_ptr,
                            devA, m,
                            devB, k,
                            beta_ptr,
                            devC, m,
                            num_moduli, fastmode, work);
    }

    cudaDeviceSynchronize();
    plhs[0] = mxGPUCreateMxArrayOnGPU(C);

    cublasDestroy(ch);
    cudaFree(work);
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
    mxGPUDestroyGPUArray(C);
}
