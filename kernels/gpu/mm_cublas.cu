// ====================================================================
// FILE: kernels/gpu/mm_cublas.cu (The Final, Textbook-Correct Fix)
// ====================================================================
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ad/kernels_api.hpp"
#include <cstdio>
#include <assert.h>
#include <iostream>

#define APA(act, x) \
    (act == 1.f     ? fmaxf(x, 0.f) : \
     act == 2.f  ? (1.f / (1.f + __expf(-x))) : \
     act == 3.f ? log1pf(__expf(x)) : \
     act == 4.f     ? tanhf(x) : x) 

// template <ActType A> struct Act;
// template <> struct Act<ActType::ReLU>     { __device__ static float apply(float x){return fmaxf(x,0.f);} };
// template <> struct Act<ActType::Sigmoid>  { __device__ static float apply(float x){return 1.f/(1.f+__expf(-x));} };
// template <> struct Act<ActType::Softplus> { __device__ static float apply(float x){return log1pf(__expf(x));} };
// template <> struct Act<ActType::Tanh>     { __device__ static float apply(float x){return tanhf(x);} };

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
void gemm_cuda(const float* A, const float* B, float* C, float* E,
            int M, int K, int N, ag_cuda_stream_t s) {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    cublasSetStream(handle, (cudaStream_t)s);

    const float alpha = 1.f, beta = 1.f;
        cudaMemcpy(E, C, M*N*sizeof(float), cudaMemcpyDeviceToDevice);


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
        E, N);              // C(M,N) -> ldc is cols = N
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
                    int M, int N, int K, ag_cuda_stream_t s)
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




void linear_cuda(const float* A, const float* B, float* C, float* E,
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

    cudaMemcpy(E, C, M*N*sizeof(float), cudaMemcpyDeviceToDevice);

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
    E, N);        // C (M,N), ldc = N

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
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &alpha, gy, N, A, K, &beta, gB, N);
dim3 blocks( (unsigned int)(((M*K)+ 255) / 256) );
            k_vjp_c_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gC, gy, (M*K));

}



// --------------------------------------------
// Flash Attention Forward Kernel (Fixed Logic)
// --------------------------------------------
__global__ void flash_forward_kernele(
    const float* Q, const float* K, const float* V, float* O,
    const int N, const int d,
    const int Tc, const int Tr, const int Bc, const int Br,
    float* l, float* m, const float softmax_scale)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index
    int i = blockIdx.z;
    int j = threadIdx.y;

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // per batch+head
    int lm_offset  = (bx * gridDim.y * N) + (by * N);

    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S  = &sram[tile_size * 3]; // S has shape [Br x Bc]

        int row_idx = i * Br + tx;
        if (row_idx < N) {

        // load Qi
        for (int x = 0; x < d; x++) {
            Qi[tx * d + x] = Q[qkv_offset + row_idx * d + x];
        }

        float row_m_prev = -INFINITY;
        float row_l_prev = 0.0f;
        if (m && l && i < Tr) {
            // if we already processed previous tiles
            row_m_prev = m[lm_offset + row_idx];
            row_l_prev = l[lm_offset + row_idx];
        }

        float row_m_new = row_m_prev;
        float row_l_new = row_l_prev;

        // process each K/V tile
        if (j < Tc) {
            int col_base = j * Bc;
            __syncthreads();

            // Load K/V to SRAM
            if (col_base + tx < N) {
                for (int x = 0; x < d; x++) {
                    Kj[tx * d + x] = K[qkv_offset + (col_base + tx) * d + x];
                    Vj[tx * d + x] = V[qkv_offset + (col_base + tx) * d + x];
                }
            } else {
                for (int x = 0; x < d; x++) {
                    Kj[tx * d + x] = 0.0f;
                    Vj[tx * d + x] = 0.0f;
                }
            }
            //printf("wefgrhy46k");

            __syncthreads();

            // compute attention scores S = QK^T
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0.0f;
                for (int x = 0; x < d; x++) {
                    sum += Qi[tx * d + x] * Kj[y * d + x];
                }
                sum *= softmax_scale;
                S[tx * Bc + y] = sum;
                if (sum > row_m) row_m = sum;
            }

            // compute P = exp(S - row_m)
            float row_l = 0.0f;
            for (int y = 0; y < Bc; y++) {
                float val = __expf(S[tx * Bc + y] - row_m);
                S[tx * Bc + y] = val;
                row_l += val;
            }

            // combine with running max/sum
            float new_m = fmaxf(row_m_prev, row_m);
            float new_l = __expf(row_m_prev - new_m) * row_l_prev +
                          __expf(row_m - new_m) * row_l;

            // accumulate O
            for (int x = 0; x < d; x++) {
                float pv = 0.0f;
                for (int y = 0; y < Bc; y++) {
                    pv += S[tx * Bc + y] * Vj[y * d + x];
                }

                float old_O = (row_l_prev > 0)
                    ? O[qkv_offset + row_idx * d + x]
                    : 0.0f;

                float new_O = (1.0f / new_l) *
                              ( __expf(row_m_prev - new_m) * row_l_prev * old_O +
                                __expf(row_m - new_m) * pv );

                O[qkv_offset + row_idx * d + x] = new_O;
            }

            row_m_prev = new_m;
            row_l_prev = new_l;
            row_m_new  = new_m;
            row_l_new  = new_l;
        }

        // store back updated m and l
        m[lm_offset + row_idx] = row_m_new;
        l[lm_offset + row_idx] = row_l_new;
        __syncthreads();
    }
}

// --------------------------------------------
// Host Launcher
// --------------------------------------------
void run_flash_forward(const float* Q, const float* K, const float* V, float* O,
                       int B, int nh, int N, int d, ag_cuda_stream_t s) {
    const int Bc = 32, Br = 32;
    const int Tc = (N + Bc - 1) / Bc;
    const int Tr = (N + Br - 1) / Br;
    const float softmax_scale = 1.0f / sqrtf((float)d);

    size_t qkv_size = B * nh * N * d * sizeof(float);
    size_t lm_size  = B * nh * N * sizeof(float);

    float *d_l, *d_m;

    cudaMalloc(&d_l, lm_size);
    cudaMalloc(&d_m, lm_size);


    cudaMemset(d_l, 0, lm_size);
    cudaMemset(d_m, -INFINITY, lm_size); // init to -inf handled inside kernel

    dim3 grid_dim(B, nh, Tr);
    dim3 block_dim(Br, Tc);
    int shared_mem = (3 * Bc * d + Br * Bc) * sizeof(float);

    flash_forward_kernele<<<grid_dim, block_dim, shared_mem, (cudaStream_t)s>>>(
        Q, K, V, O,
        N, d, Tc, Tr, Bc, Br, d_l, d_m, softmax_scale
    );
    cudaDeviceSynchronize();


}




// --------------------------------------------
// Flash Attention Forward Kernel (Fixed Logic)
// --------------------------------------------
__global__ void flash_aliforward_kernele(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d, int Bc, int Br,
    float softmax_scale){
    extern __shared__ float smem[]; // layout explained below
    // smem layout:
    // Qi: Br * d
    // Kj: Bc * d
    // Vj: Bc * d
    // P : Br * Bc

    float* Qi = smem;                                 // Br * d
    float* Kj = Qi + Br * d;                          // Bc * d
    float* Vj = Kj + Bc * d;                          // Bc * d
    float* P  = Vj + Bc * d;                          // Br * Bc

    int batch  = blockIdx.x;
    int head   = blockIdx.y;
    int tile_i = blockIdx.z; // i tile index

    int tx = threadIdx.x; // 0..Br-1

    int qkv_offset = ((batch * gridDim.y) + head) * N * d;

    int row = tile_i * Br + tx;
    if (row >= N) return;

    // load Q[row] into Qi[tx*d + ...]
    for (int x = 0; x < d; ++x) {
        Qi[tx * d + x] = Q[qkv_offset + row * d + x];
    }

    // prefix stats
    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    // ALiBi slope: correct formula using nh = gridDim.y
    float slope_start = powf(2.0f, -8.0f / (float)gridDim.y);
    float slope = powf(slope_start, (float)head);

    int Tc = (N + Bc - 1) / Bc;

    // For each K/V tile (sequentially!)
    for (int t = 0; t < Tc; ++t) {
        int col_base = t * Bc;

        __syncthreads();

        // load K/V tile into shared mem with strided loads so all entries are filled
        // each thread loads multiple rows of K/V: idx ranges over [0, Bc)
        for (int idx = tx; idx < Bc; idx += Br) {
            int col = col_base + idx;
            if (col < N) {
                for (int x = 0; x < d; ++x) {
                    Kj[idx * d + x] = K[qkv_offset + col * d + x];
                    Vj[idx * d + x] = V[qkv_offset + col * d + x];
                }
            } else {
                // pad zeros for out-of-range columns
                for (int x = 0; x < d; ++x) {
                    Kj[idx * d + x] = 0.0f;
                    Vj[idx * d + x] = 0.0f;
                }
            }
        }
        __syncthreads();

        // compute Q * K^T for this tile, store in P, track max
        float m_curr = -INFINITY;
        for (int y = 0; y < Bc; ++y) {
            int j_pos = col_base + y;
            float s = 0.0f;
            // dot product Qi[tx,:] . Kj[y,:]
            for (int x = 0; x < d; ++x) {
                s += Qi[tx * d + x] * Kj[y * d + x];
            }
            s *= softmax_scale;

            // causal mask + ALiBi
            if (j_pos > row) {
                s = -INFINITY;
            } else {
                s += (float)(row - j_pos) * slope;
            }

            P[tx * Bc + y] = s;
            if (s > m_curr) m_curr = s;
        }

        // compute unnormalized probs for this tile and l_curr
        float l_curr = 0.0f;
        // If m_curr == -INFINITY (all masked) then all exps are 0 and l_curr stays 0
        if (m_curr > -INFINITY/2.0f) {
            for (int y = 0; y < Bc; ++y) {
                float v = __expf(P[tx * Bc + y] - m_curr);
                P[tx * Bc + y] = v;
                l_curr += v;
            }
        } else {
            // all masked -> set P entries to 0
            for (int y = 0; y < Bc; ++y) P[tx * Bc + y] = 0.0f;
            l_curr = 0.0f;
        }

        // prefix-merge in a numerically stable way
        float m_new = fmaxf(m_prev, m_curr);
        // handle m_prev == -INF case: exp(-INF - m_new) -> 0, fine
        float l_new;
        if (l_prev == 0.0f && l_curr == 0.0f) {
            l_new = 0.0f;
        } else {
            l_new = ( (l_prev == 0.0f) ? 0.0f : __expf(m_prev - m_new) * l_prev )
                  + ( (l_curr == 0.0f) ? 0.0f : __expf(m_curr - m_new) * l_curr );
        }

        // accumulate output for this tile: O_new = (exp(m_prev-m_new)*l_prev*old_O + exp(m_curr-m_new)*pv) / l_new
        if (l_new == 0.0f) {
            // nothing to accumulate (all masked). Keep O as-is or zero (O should be zero-initialized).
            // For safety, write zero (this avoids any 0/0).
            for (int x = 0; x < d; ++x) {
                O[qkv_offset + row * d + x] = 0.0f;
            }
        } else {
            float coef_prev = (l_prev == 0.0f) ? 0.0f : __expf(m_prev - m_new) * l_prev / l_new;
            float coef_curr = (l_curr == 0.0f) ? 0.0f : __expf(m_curr - m_new) * 1.0f / l_new; // multiply pv by l_curr inside acc
            // compute pv = sum_y P[y] * Vj[y,:]  (note P[y] is already exp(S - m_curr))
            for (int x = 0; x < d; ++x) {
                float pv = 0.0f;
                for (int y = 0; y < Bc; ++y) {
                    pv += P[tx * Bc + y] * Vj[y * d + x];
                }
                float old = (coef_prev == 0.0f) ? 0.0f : O[qkv_offset + row * d + x];
                float newO = coef_prev * old + coef_curr * pv * l_curr; // note coef_curr * pv * l_curr == exp(m_curr-m_new)*pv / l_new
                O[qkv_offset + row * d + x] = newO;
            }
        }

        // update prefix stats
        m_prev = m_new;
        l_prev = l_new;
    } // for tiles
}

// --------------------------------------------
// Host Launcher
// --------------------------------------------
void run_flash_aliforward(const float* Q, const float* K, const float* V, float* O,
                       int B, int H, int N, int d, ag_cuda_stream_t s) {
    const int Br = 32;
    const int Bc = 32;
    float softmax_scale = 1.0f / sqrtf((float)d);

    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(Br);

    size_t qkv_size = (size_t)B * H * N * d * sizeof(float);

    // ZERO O before running (VERY IMPORTANT)
    cudaMemsetAsync(O, 0, qkv_size, (cudaStream_t)s);

    int smem = (Br * d + Bc * d + Bc * d + Br * Bc) * sizeof(float);
    flash_aliforward_kernele<<<grid, block, smem, (cudaStream_t)s>>>(
        Q, K, V, O, N, d, Bc, Br, softmax_scale
    );


}

__global__ void flash_reluforward_kernele(
    const float* Q, const float* K, const float* V, float* O,
    const int N, const int d,
    const int Tc, const int Tr, const int Bc, const int Br,
    float* l, float* m, const float softmax_scale)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index
    int i = blockIdx.z;
    int j = threadIdx.y;

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // per batch+head
    int lm_offset  = (bx * gridDim.y * N) + (by * N);

    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S  = &sram[tile_size * 3]; // S has shape [Br x Bc]

        int row_idx = i * Br + tx;
        if (row_idx < N) {

        // load Qi
        for (int x = 0; x < d; x++) {
            Qi[tx * d + x] = Q[qkv_offset + row_idx * d + x];
        }

        float row_m_prev = -INFINITY;
        float row_l_prev = 0.0f;
        if (m && l && i < Tr) {
            // if we already processed previous tiles
            row_m_prev = m[lm_offset + row_idx];
            row_l_prev = l[lm_offset + row_idx];
        }

        float row_m_new = row_m_prev;
        float row_l_new = row_l_prev;

        // process each K/V tile
        if (j < Tc) {
            int col_base = j * Bc;
            __syncthreads();

            // Load K/V to SRAM
            if (col_base + tx < N) {
                for (int x = 0; x < d; x++) {
                    Kj[tx * d + x] = K[qkv_offset + (col_base + tx) * d + x];
                    Vj[tx * d + x] = V[qkv_offset + (col_base + tx) * d + x];
                }
            } else {
                for (int x = 0; x < d; x++) {
                    Kj[tx * d + x] = 0.0f;
                    Vj[tx * d + x] = 0.0f;
                }
            }
            printf("qq");

            __syncthreads();

            // compute attention scores S = QK^T
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0.0f;
                for (int x = 0; x < d; x++) {
                    sum += Qi[tx * d + x] * Kj[y * d + x];
                }
                sum *= softmax_scale; // APPLY SCALING HERE
                S[tx * Bc + y] = sum;
                if (sum > row_m) row_m = sum;
            }

       // compute P = exp(S - row_m)
            float row_l = 0.0f;
for (int y = 0; y < Bc; y++) {
    float val = fmaxf(S[tx * Bc + y], 0.0f);  // ReLU activation
    S[tx * Bc + y] = val;
    row_l += val;
}

            // combine with running max/sum
            float new_m = fmaxf(row_m_prev, row_m);
            float new_l = row_l_prev + row_l;  // Just sum the normalizers

            // accumulate O
            // During accumulation (lines 84-96), CHANGE TO:
            // accumulate O
            for (int x = 0; x < d; x++) {
                float pv = 0.0f;
                for (int y = 0; y < Bc; y++) {
                    pv += S[tx * Bc + y] * Vj[y * d + x];
                }

                float old_O = (row_l_prev > 0)
                    ? O[qkv_offset + row_idx * d + x]
                    : 0.0f;


    O[qkv_offset + row_idx * d + x] = old_O + pv;  // Just accumulate, don't normalize yet
}

            row_m_prev = new_m;
            row_l_prev = new_l;
            row_m_new  = new_m;
            row_l_new  = new_l;
        }

        // store back updated m and l
        m[lm_offset + row_idx] = row_m_new;
        l[lm_offset + row_idx] = row_l_new;
        __syncthreads();
    }
}

// --------------------------------------------
// Host Launcher
// --------------------------------------------
void run_flashrelu_forward(const float* Q, const float* K, const float* V, float* O,
                       int B, int nh, int N, int d, ag_cuda_stream_t s) {
    const int Bc = 32, Br = 32;
    const int Tc = (N + Bc - 1) / Bc;
    const int Tr = (N + Br - 1) / Br;
    const float softmax_scale = 1.0f / sqrtf((float)d);

    size_t qkv_size = B * nh * N * d * sizeof(float);
    size_t lm_size  = B * nh * N * sizeof(float);

    float *d_l, *d_m;

    cudaMalloc(&d_l, lm_size);
    cudaMalloc(&d_m, lm_size);


    cudaMemset(d_l, 0, lm_size);
    cudaMemset(d_m, 0, lm_size); // init to -inf handled inside kernel

    dim3 grid_dim(B, nh, Tr);
    dim3 block_dim(Br, Tc);
    int shared_mem = (3 * Bc * d + Br * Bc) * sizeof(float);

    flash_reluforward_kernele<<<grid_dim, block_dim, shared_mem, (cudaStream_t)s>>>(
        Q, K, V, O,
        N, d, Tc, Tr, Bc, Br, d_l, d_m, softmax_scale
    );
    cudaDeviceSynchronize();


}

__device__ __forceinline__ float sigmoidf(float x) {
  return 1.f / (1.f + __expf(-x));
}

__global__ void flash_sigforward_kernele(
    const float* Q, const float* K, const float* V, float* O,
    const int N, const int d,
    const int Tc, const int Tr, const int Bc, const int Br,
    float* l, float* m, const float softmax_scale)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index
    int i = blockIdx.z;
    int j = threadIdx.y;

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // per batch+head
    int lm_offset  = (bx * gridDim.y * N) + (by * N);

    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S  = &sram[tile_size * 3]; // S has shape [Br x Bc]

        int row_idx = i * Br + tx;
        if (row_idx < N) {

        // load Qi
        for (int x = 0; x < d; x++) {
            Qi[tx * d + x] = Q[qkv_offset + row_idx * d + x];
        }

        float row_m_prev = -INFINITY;
        float row_l_prev = 0.0f;
        if (m && l && i < Tr) {
            // if we already processed previous tiles
            row_m_prev = m[lm_offset + row_idx];
            row_l_prev = l[lm_offset + row_idx];
        }

        float row_m_new = row_m_prev;
        float row_l_new = row_l_prev;

        // process each K/V tile
        if (j < Tc) {
            int col_base = j * Bc;
            __syncthreads();

            // Load K/V to SRAM
            if (col_base + tx < N) {
                for (int x = 0; x < d; x++) {
                    Kj[tx * d + x] = K[qkv_offset + (col_base + tx) * d + x];
                    Vj[tx * d + x] = V[qkv_offset + (col_base + tx) * d + x];
                }
            } else {
                for (int x = 0; x < d; x++) {
                    Kj[tx * d + x] = 0.0f;
                    Vj[tx * d + x] = 0.0f;
                }
            }
            printf("qq");

            __syncthreads();

            // compute attention scores S = QK^T
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0.0f;
                for (int x = 0; x < d; x++) {
                    sum += Qi[tx * d + x] * Kj[y * d + x];
                }
                sum *= softmax_scale; // APPLY SCALING HERE
                S[tx * Bc + y] = sum;
                if (sum > row_m) row_m = sum;
            }

       // compute P = exp(S - row_m)
            float row_l = 0.0f;
for (int y = 0; y < Bc; y++) {
    float val = sigmoidf(S[tx * Bc + y]);  // ReLU activation
    S[tx * Bc + y] = val;
    row_l += val;
}

            // combine with running max/sum
            float new_m = sigmoidf(row_m_prev)+row_m;
            float new_l = row_l_prev + row_l;  // Just sum the normalizers

            // accumulate O
            // During accumulation (lines 84-96), CHANGE TO:
            // accumulate O
            for (int x = 0; x < d; x++) {
                float pv = 0.0f;
                for (int y = 0; y < Bc; y++) {
                    pv += S[tx * Bc + y] * Vj[y * d + x];
                }

                float old_O = (row_l_prev > 0)
                    ? O[qkv_offset + row_idx * d + x]
                    : 0.0f;

                    
    O[qkv_offset + row_idx * d + x] = old_O + pv;  // Just accumulate, don't normalize yet
}

            row_m_prev = new_m;
            row_l_prev = new_l;
            row_m_new  = new_m;
            row_l_new  = new_l;
        }

        // store back updated m and l
        m[lm_offset + row_idx] = row_m_new;
        l[lm_offset + row_idx] = row_l_new;
        __syncthreads();
    }
}

// --------------------------------------------
// Host Launcher
// --------------------------------------------
void run_flashsig_forward(const float* Q, const float* K, const float* V, float* O,
                       int B, int nh, int N, int d, ag_cuda_stream_t s) {
    const int Bc = 32, Br = 32;
    const int Tc = (N + Bc - 1) / Bc;
    const int Tr = (N + Br - 1) / Br;
    const float softmax_scale = 1.0f / sqrtf((float)d);

    size_t qkv_size = B * nh * N * d * sizeof(float);
    size_t lm_size  = B * nh * N * sizeof(float);

    float *d_l, *d_m;

    cudaMalloc(&d_l, lm_size);
    cudaMalloc(&d_m, lm_size);


    cudaMemset(d_l, 0, lm_size);
    cudaMemset(d_m, 0, lm_size); // init to -inf handled inside kernel

    dim3 grid_dim(B, nh, Tr);
    dim3 block_dim(Br, Tc);
    int shared_mem = (3 * Bc * d + Br * Bc) * sizeof(float);

    flash_sigforward_kernele<<<grid_dim, block_dim, shared_mem, (cudaStream_t)s>>>(
        Q, K, V, O,
        N, d, Tc, Tr, Bc, Br, d_l, d_m, softmax_scale
    );
    cudaDeviceSynchronize();


}




// --------------- Generic FlexAttention Kernel -----------------
//template <typename Activation>
__global__ void flash_flexforward_kerneleoo(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, int Tc, int Tr, int Bc, int Br,
    float* l, float* m, const float * act, float softmax_scale)
{
        int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index
    int i = blockIdx.z;
    int j = threadIdx.y;

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // per batch+head
    int lm_offset  = (bx * gridDim.y * N) + (by * N);

    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S  = &sram[tile_size * 3]; // S has shape [Br x Bc]

        int row_idx = i * Br + tx;
        if (row_idx < N) {

        // load Qi
        for (int x = 0; x < d; x++) {
            Qi[tx * d + x] = Q[qkv_offset + row_idx * d + x];
        }

        float row_m_prev = -INFINITY;
        float row_l_prev = 0.0f;
        if (m && l && i < Tr) {
            // if we already processed previous tiles
            row_m_prev = m[lm_offset + row_idx];
            row_l_prev = l[lm_offset + row_idx];
        }

        float row_m_new = row_m_prev;
        float row_l_new = row_l_prev;

        // process each K/V tile
        if (j < Tc) {
            int col_base = j * Bc;
            __syncthreads();

            // Load K/V to SRAM
            if (col_base + tx < N) {
                for (int x = 0; x < d; x++) {
                    Kj[tx * d + x] = K[qkv_offset + (col_base + tx) * d + x];
                    Vj[tx * d + x] = V[qkv_offset + (col_base + tx) * d + x];
                }
            } else {
                for (int x = 0; x < d; x++) {
                    Kj[tx * d + x] = 0.0f;
                    Vj[tx * d + x] = 0.0f;
                }
            }
            printf("qq");

            __syncthreads();

            // compute attention scores S = QK^T
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0.0f;
                for (int x = 0; x < d; x++) {
                    sum += Qi[tx * d + x] * Kj[y * d + x];
                }
                sum *= softmax_scale; // APPLY SCALING HERE
                S[tx * Bc + y] = sum;
                if (sum > row_m) row_m = sum;
            }

       // compute P = exp(S - row_m)
            float row_l = 0.0f;
for (int y = 0; y < Bc; y++) {
    float val = APA(*act,S[tx * Bc + y]);  // ReLU activation
    S[tx * Bc + y] = val;
    row_l += val;
}

            // combine with running max/sum
            float new_m = APA(*act,row_m_prev)+row_m;
            float new_l = row_l_prev + row_l;  // Just sum the normalizers

            // accumulate O
            // During accumulation (lines 84-96), CHANGE TO:
            // accumulate O
            for (int x = 0; x < d; x++) {
                float pv = 0.0f;
                for (int y = 0; y < Bc; y++) {
                    pv += S[tx * Bc + y] * Vj[y * d + x];
                }

                float old_O = (row_l_prev > 0)
                    ? O[qkv_offset + row_idx * d + x]
                    : 0.0f;

                    
    O[qkv_offset + row_idx * d + x] = old_O + pv;  // Just accumulate, don't normalize yet
}

            row_m_prev = new_m;
            row_l_prev = new_l;
            row_m_new  = new_m;
            row_l_new  = new_l;
        }

        // store back updated m and l
        m[lm_offset + row_idx] = row_m_new;
        l[lm_offset + row_idx] = row_l_new;
        __syncthreads();
    }
}



// --------------- Host launcher macro -----------------
void run_flashflex_forwardz(const float* Q, const float* K, const float* V, float* O,
                                     int B, int nh, int N, int d, const float* act, cudaStream_t s)
{



    const int Bc = 32, Br = 32;
    const int Tc = (N + Bc - 1) / Bc;
    const int Tr = (N + Br - 1) / Br;
    const float softmax_scale = 1.0f / sqrtf((float)d);

    size_t qkv_size = B * nh * N * d * sizeof(float);
    size_t lm_size  = B * nh * N * sizeof(float);

    float *d_l, *d_m;

    cudaMalloc(&d_l, lm_size);
    cudaMalloc(&d_m, lm_size);


    cudaMemset(d_l, 0, lm_size);
    cudaMemset(d_m, 0, lm_size); // init to -inf handled inside kernel

    float* d_act;
cudaMalloc(&d_act, N * sizeof(float));
cudaMemcpy(d_act, act, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_dim(B, nh, Tr);
    dim3 block_dim(Br, Tc);
    int shared_mem = (3 * Bc * d + Br * Bc) * sizeof(float);

        flash_flexforward_kerneleoo<<<grid_dim, block_dim, shared_mem, (cudaStream_t)s>>>(
        Q, K, V, O,
        N, d, Tc, Tr, Bc, Br, d_l, d_m, d_act, softmax_scale
    );
    cudaDeviceSynchronize();
    cudaFree(d_act);
    
}




__global__ void sigmoidline(float * val, float * sal, float j) {
    // Perform warp-level reduction for max operation
    int i =  blockDim.x*blockDim.x + threadIdx.x ;
    if(i<j)
     val[i] = (val[i] * sal[i] )/ (1.f + __expf(-val[i]));

}


                        
void swiglu_cuda(const float* A, const float* B, const float* D, const float* F, const float* G, float* C, float* E,
                 int M, int K, int N,// int W, // W is used for, 
                 ag_cuda_stream_t s) {
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

    cudaMemcpy(E, D, M*N*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(C, G, M*N*sizeof(float), cudaMemcpyDeviceToDevice);

    // Explanation:
    // We want E = D + A @ B^T (row-major)
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
    E, N);        // C (M,N), ldc = N

    // We want C = G + A @ F^T (row-major)

cublasSgemm(handle,
    CUBLAS_OP_T,  // <- transpose B
    CUBLAS_OP_N,  // <- don't transpose A
    N, M, K,      // m=N, n=M, k=K
    &alpha,
    F, K,         // B is (N, K) so lda = K
    A, K,         // A is (M, K) so ldb = K
    &beta,
    C, N);        // C (M,N), ldc = N
int64_t blocks((unsigned int)(((M*N) + 255) / 256));
    sigmoidline<<<blocks, 256, 0, (cudaStream_t)s>>>(E, C, M*N);


    //   B is row-major (N x K), but used as (K x N) in column-major view -> lda = K
    //   A is row-major (M x K), used as (K x M) when transposed -> ldb = K
    //   C is row-major (M x N), used as (N x M) in column-major view -> ldc = N

}