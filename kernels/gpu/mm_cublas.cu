// ====================================================================
// FILE: kernels/gpu/mm_cublas.cu (The Final, Textbook-Correct Fix)
// ====================================================================
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ad/kernels_api.hpp"
#include <cstdio>
#include <assert.h>

// --- Forward Pass: C = A @ B ---
// A is (M, K), B is (K, N), C is (M, N). All are row-major.
void mm_cuda(const float* A, const float* B, float* C,
            int M, int K, int N, ag_cuda_stream_t s) {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    cublasSetStream(handle, (cudaStream_t)s);

    const float alpha = 1.f, beta = 0.f;

    // Row-major C(M,N) = A(M,K) @ B(K,N) is equivalent to
    // column-major C_t(N,M) = B_t(N,K) @ A_t(K,M).
    // We pass B as the first matrix and A as the second, with NO transpose flags.
    // API: cublasSgemm(handle, transa, transb, m, n, k, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc)
    cublasSgemm(handle,
        CUBLAS_OP_N,        // transa on B
        CUBLAS_OP_N,        // transb on A
        N, M, K,            // m=N, n=M, k=K
        &alpha,
        B, N,               // B(K,N) -> lda is cols = N
        A, K,               // A(M,K) -> ldb is cols = K
        &beta,
        C, N);              // C(M,N) -> ldc is cols = N
}

// --- Backward Pass (VJP) ---
void vjp_matmul_cuda(float* gA, float* gB, const float* gy,
                    const float* A, const float* B,
                    int M, int K, int N, ag_cuda_stream_t s)
{
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    cublasSetStream(handle, (cudaStream_t)s);
    const float alpha = 1.f, beta = 0.f;

    // gA(M,K) = gy(M,N) @ B^T(N,K)
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &alpha, B, N, gy, N, &beta, gA, K);

    // gB(K,N) = A^T(K,M) @ gy(M,N)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &alpha, gy, N, A, K, &beta, gB, N);
}


// --- Forward Pass: C = A @ B ---
// A is (M, K), B is (K, N), C is (M, N). All are row-major.
void gemm_cuda(const float* A, const float* B, float* C,
            int M, int K, int N, ag_cuda_stream_t s) {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    cublasSetStream(handle, (cudaStream_t)s);

    const float alpha = 1.f, beta = 1.f;

    // Row-major C(M,N) = A(M,K) @ B(K,N) is equivalent to
    // column-major C_t(N,M) = B_t(N,K) @ A_t(K,M).
    // We pass B as the first matrix and A as the second, with NO transpose flags.
    // API: cublasSgemm(handle, transa, transb, m, n, k, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc)
    cublasSgemm(handle,
        CUBLAS_OP_N,        // transa on B
        CUBLAS_OP_N,        // transb on A
        N, M, K,            // m=N, n=M, k=K
        &alpha,
        B, N,               // B(K,N) -> lda is cols = N
        A, K,               // A(M,K) -> ldb is cols = K
        &beta,
        C, N);              // C(M,N) -> ldc is cols = N
}

__global__ void k_vjp_c_accum(float* gC, const float* gy, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&gC[i], gy[i]);
    }
}

// --- Backward Pass (VJP) ---
void vjp_gemm_cuda(float* gA, float* gB, float* gC, const float* gy,
                    const float* A, const float* B, const float* C,
                    int M, int K, int N, ag_cuda_stream_t s)
{
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    cublasSetStream(handle, (cudaStream_t)s);
    const float alpha = 1.f, beta = 1.f;

    // gA(M,K) = gy(M,N) @ B^T(N,K)
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &alpha, B, N, gy, N, &beta, gA, K);

    // gB(K,N) = A^T(K,M) @ gy(M,N)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &alpha, gy, N, A, K, &beta, gB, N);

dim3 blocks( (unsigned int)(((M*K)+ 255) / 256) );
            k_vjp_c_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gC, gy, (M*K));

}




void linear_cuda(const float* A, const float* B, float* C,
                 int M, int K, int N, ag_cuda_stream_t s) {
    // Semantics: A: (M,K), B: (N,K)  -> compute C += A @ B^T  (C is MxN row-major)
    assert(A && B && C);
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "linear_cuda: cublasCreate failed\n");
            return;
        }
    }
    cublasSetStream(handle, s);

    const float alpha = 1.0f;
    const float beta  = 1.0f; // match "GEMM worked as C += A@B", use 1.0f for accumulation

    // Explanation:
    // We want C = C + A @ B^T (row-major)
    // Map to column-major cuBLAS: compute C^T = B * A^T
    // call: cublasSgemm(handle, opA=CUBLAS_OP_N (B), opB=CUBLAS_OP_T (A), m=N, n=M, k=K, ...)
    // Leading dims:
    //   B is row-major (N x K), but used as (K x N) in column-major view -> lda = K
    //   A is row-major (M x K), used as (K x M) when transposed -> ldb = K
    //   C is row-major (M x N), used as (N x M) in column-major view -> ldc = N
cublasSgemm(handle,
    CUBLAS_OP_T,  // <- transpose B
    CUBLAS_OP_N,  // <- don't transpose A
    N, M, K,      // m=N, n=M, k=K
    &alpha,
    B, K,         // B is (N, K) so lda = K
    A, K,         // A is (M, K) so ldb = K
    &beta,
    C, N);        // C (M,N), ldc = N
}



// --- Backward Pass (VJP) ---
void vjp_linear_cuda(float* gA, float* gB, float* gC, const float* gy,
                    const float* A, const float* B, const float* C,
                    int M, int K, int N, ag_cuda_stream_t s)
{
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    cublasSetStream(handle, (cudaStream_t)s);
    const float alpha = 1.f, beta = 1.f;

    // gA(M,K) = gy(M,N) @ B(N,K)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, B, K, gy, N, &beta, gA, K);

    // gB(N,K) = gy_t(N,M) @ A(M,K) 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, N, M, &alpha, A, K, gy, N, &beta, gB, K);

dim3 blocks( (unsigned int)(((M*K)+ 255) / 256) );
            k_vjp_c_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gC, gy, (M*K));

}