// kernels/gpu/eltwise.cu
#include <cuda_runtime.h>
#include <cstdint>
#include "ad/kernels_api.hpp"
#include <math_constants.h>
#include <cstdio>
#include <cfloat>


__device__ __forceinline__ float sigmoidf(float x) {
  return 1.f / (1.f + __expf(-x));
}

__device__ __forceinline__ float tanhf_fast(float x) {
  return tanhf(x);
}

__device__ __forceinline__ float gelu_func(float x) {
  const float k0 = 0.7978845608f; // sqrt(2/pi)
  const float k1 = 0.044715f;
  return 0.5f * x * (1.f + tanhf(k0 * (x + k1 * x * x * x)));
}

__device__ __forceinline__ float mish_func(float x) {
  return x * tanhf(__logf(1.f + __expf(x)));
}

// =====================================================
// Elementwise Kernels
// =====================================================

__global__ void k_add(const float* __restrict__ a, const float* __restrict__ b,
                      float* __restrict__ c, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void k_sub(const float* __restrict__ a, const float* __restrict__ b,
                      float* __restrict__ c, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] - b[i];
}

__global__ void k_mul(const float* __restrict__ a, const float* __restrict__ b,
                      float* __restrict__ c, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] * b[i];
}

__global__ void k_div(const float* __restrict__ a, const float* __restrict__ b,
                      float* __restrict__ c, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] / b[i];
}

__global__ void k_pow(const float* __restrict__ a, const float* __restrict__ b,
                      float* __restrict__ c, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = __powf(a[i], b[i]);
}

__global__ void k_square(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * x[i];
}

__global__ void k_neg(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = -x[i];
}

__global__ void k_clip(const float* __restrict__ x, float* __restrict__ y,
                       float minv, float maxv, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = (x[i]>maxv) ? maxv : (   x[i]<minv ? minv : x[i]            );
  }
}

// =====================================================
// Activations
// =====================================================

__global__ void k_relu(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] > 0.f ? x[i] : 0.f;
  }
}

__global__ void k_relumask(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] > 0.f ? 1.f : 0.f;
  }
}

__global__ void k_leaky_relu(const float* __restrict__ x, float* __restrict__ y,
                             float slope, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] > 0.f ? x[i] : slope * x[i];
  }
}

__global__ void k_sign(const float* __restrict__ x, float* __restrict__ y,
                             int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] > 0.f ? 1.f : -1.f;
  }
}

__global__ void k_gelu(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = gelu_func(x[i]);
}

__global__ void k_silu(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] * sigmoidf(x[i]);
  }
}

__global__ void k_mish(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = mish_func(x[i]);
}

__global__ void k_tanh(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = tanhf_fast(x[i]);
}

__global__ void k_sigmoid(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = sigmoidf(x[i]);
}

__global__ void k_hard_sigmoid(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = 0.2f * x[i] + 0.5f;
        y[i] = (v>1.f) ? 1.f : (   v<0.f ? 0.f : v           );

  }
}

__global__ void k_hard_swish(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float hs = 0.2f * x[i] + 0.5f;
    y[i] = x[i] * ((hs>1.f) ? 1.f : (   hs<0.f ? 0.f : hs           ))    ;
  }
}

__global__ void k_exp(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __expf(x[i]);
}

__global__ void k_log(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __logf(x[i]);
}

__global__ void k_cos(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = cos(x[i]);
}

__global__ void k_gcu(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i]*cos(x[i]);
}

__global__ void k_gauss(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = exp(-x[i]*x[i]);
}

__global__ void k_parcon(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i]*(2-x[i]);
}

__global__ void k_lisht(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i]*tanh(x[i]);
}

__global__ void k_dyntanh(const float* __restrict__ x, float a, float  b, float  g, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (g*tanh(x[i]*a))+b;
}

__global__ void k_reci(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (1.f/x[i]);
}

__global__ void k_sin(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = sin(x[i]);
}

__global__ void k_cosh(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = cosh(x[i]);
}

__global__ void k_sinh(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = sinh(x[i]);
}

__global__ void k_sqrt(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = sqrt(x[i]);
}

__global__ void k_softplus(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __logf(1.0f + __expf(x[i]));


}






__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // supports up to 1024 threads
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    // reduce within warp
    val = warpReduceSum(val);

    // write warp sum to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // read warp sums and reduce them in warp 0
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void k_sumall(const float* __restrict__ g_idata,
                         float* __restrict__ g_odata,
                         int N)
{
    float sum = 0.0f;

    // grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        sum += g_idata[i];

    // block-wide reduction
    sum = blockReduceSum(sum);

    // one thread per block writes result
    if (threadIdx.x == 0)
        atomicAdd(g_odata, sum);
}

__global__ void k_rowsum(const float* __restrict__ input,
                         float* __restrict__ output,
                         int rows, int cols)
{
     // Each block handles multiple rows
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;  // Ensure we're within bounds

    // Initialize sum for this thread
    float sum = 0.0f;

    // Each thread will sum over all columns in its row
    for (int col = 0; col < cols; col++) {
        int idx = row * cols + col;
        sum += input[idx];
    }

    // Atomic add to accumulate the row sum in global memory
    atomicAdd(&output[row], sum);
}


// __inline__ __device__ float warpReduceMax(float val) {
//     // Perform warp-level reduction for max operation
//     for (int offset = warpSize / 2; offset > 0; offset /= 2) {
//         val = max(val, __shfl_down_sync(0xffffffff, val, offset));
//     }
//     return val;
// }

// __inline__ __device__ float blockReduceMax(float val) {
//     __shared__ float shared[32]; // supports up to 1024 threads

//     int lane = threadIdx.x % warpSize;
//     int wid  = threadIdx.x / warpSize;

//     // reduce within warp (max operation)
//     val = warpReduceMax(val);

//     // write warp max to shared memory
//     if (lane == 0) shared[wid] = val;
//     __syncthreads();

//     // read warp maxes and reduce them in warp 0
//     val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -FLT_MAX;  // Initialize with smallest possible value
//     if (wid == 0) val = warpReduceMax(val);
//     return val;
// }

__global__ void k_rowmax(const float* __restrict__ input,
                         float* __restrict__ output,
                         int rows, int cols)
{
    // Each block handles multiple rows (same as k_rowsum)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;  // Ensure we're within bounds
    
    // Initialize max for this thread
    float row_max = -FLT_MAX;
    
    // Each thread will find max over all columns in its row
    for (int col = 0; col < cols; col++) {
        int idx = row * cols + col;
        row_max = max(row_max, input[idx]);
    }
    
    // Write the max value for this row (no atomic needed, each thread writes to different location)
    output[row] = row_max;
}

// __inline__ __device__ void kaa_rowmax(const float* __restrict__ input,
//                          float* __restrict__ output,
//                          int rows, int cols)
// {
//     // Each block handles multiple rows (same as k_rowsum)
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row >= rows) return;  // Ensure we're within bounds
    
//     // Initialize max for this thread
//     float row_max = -FLT_MAX;
    
//     // Each thread will find max over all columns in its row
//     for (int col = 0; col < cols; col++) {
//         int idx = row * cols + col;
//         row_max = max(row_max, input[idx]);
//     }
    
//     // Write the max value for this row (no atomic needed, each thread writes to different location)
//     output[row] = row_max;
// }




// __inline__ __device__ void kaa_sumall(const float* __restrict__ g_idata,
//                          float* __restrict__ g_odata,
//                          int N)
// {
//     float sum = 0.0f;

//     // grid-stride loop
//     for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
//         sum += g_idata[i];

//     // block-wide reduction
//     sum = blockReduceSum(sum);

//     // one thread per block writes result
//     if (threadIdx.x == 0)
//         atomicAdd(g_odata, sum);
// }

// __inline__ __device__ void kaa_rowsum(const float* __restrict__ input,
//                          float* __restrict__ output,
//                          int rows, int cols)
// {
//      // Each block handles multiple rows
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row >= rows) return;  // Ensure we're within bounds

//     // Initialize sum for this thread
//     float sum = 0.0f;

//     // Each thread will sum over all columns in its row
//     for (int col = 0; col < cols; col++) {
//         int idx = row * cols + col;
//         sum += input[idx];
//     }

//     // Atomic add to accumulate the row sum in global memory
//     atomicAdd(&output[row], sum);
// }


__inline__ __device__ float kaa_rowmax(const float* input, int row, int cols) {
    float max_val = -FLT_MAX;
    for (int col = 0; col < cols; ++col)
        max_val = fmaxf(max_val, input[row * cols + col]);
    return max_val;
}

__inline__ __device__ float kaa_rowsum(const float* input, int row, int cols) {
    float sum_val = 0.0f;
    for (int col = 0; col < cols; ++col)
        sum_val += input[row * cols + col];
    return sum_val;
}

__inline__ __device__ void kaa_rowexp(const float* input, float* output, int row, int cols, float row_max) {
    for (int col = 0; col < cols; ++col)
        output[row * cols + col] = __expf(input[row * cols + col] - row_max);
}

__global__ void k_softmax(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float max_val = kaa_rowmax(input, row, cols);

    kaa_rowexp(input, output, row, cols, max_val);

    float sum_val = kaa_rowsum(output, row, cols);
    for (int col = 0; col < cols; ++col)
        output[row * cols + col] /= sum_val;
}



// =====================================================
// Launch Helpers
// =====================================================

static inline dim3 blocks_for(int64_t n, int tpb = 256) {
  return dim3(int((n + tpb - 1) / tpb));
}

// =====================================================
// CUDA Launch Functions
// =====================================================

void sumall_cuda(const float* a, float* c, int64_t n, ag_cuda_stream_t s) {
  k_sumall<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(a, c, n);
}

void rowsum_cuda(const float* a, float* c, int64_t n, int64_t w, ag_cuda_stream_t s) {
  k_rowsum<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(a, c, n, w);

//   float* output_host = (float*)malloc(n * sizeof(float));
// cudaMemcpy(output_host, c, n * sizeof(float), cudaMemcpyDeviceToHost);

// // Print results on the host
// for (int i = 0; i < n; i++) {
//     printf("Row %d sum: %f\n", i, output_host[i]);
// }

// free(output_host);
}

void softmax_cuda(const float* a, float* c, int64_t n, int64_t w, ag_cuda_stream_t s) {
  k_softmax<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(a, c, n, w);

//   float* output_host = (float*)malloc(n * sizeof(float));
// cudaMemcpy(output_host, c, n * sizeof(float), cudaMemcpyDeviceToHost);

// // Print results on the host
// for (int i = 0; i < n; i++) {
//     printf("Row %d sum: %f\n", i, output_host[i]);
// }

// free(output_host);
}


void rowmax_cuda(const float* a, float* c, int64_t n, int64_t w, ag_cuda_stream_t s) {
  k_rowmax<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(a, c, n, w);

//   float* output_host = (float*)malloc(n * sizeof(float));
// cudaMemcpy(output_host, c, n * sizeof(float), cudaMemcpyDeviceToHost);

// // Print results on the host
// for (int i = 0; i < n; i++) {
//     printf("Row %d sum: %f\n", i, output_host[i]);
// }

// free(output_host);
}


__global__ void k_mseloss(const float* __restrict__ p, const float* __restrict__ t,
                      float* __restrict__ c, int64_t n) {
    float sum = 0.0f;

    // grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x)
    {
        float diff = p[i] - t[i];
        sum += diff * diff;
    }

    // block reduce
    sum = blockReduceSum(sum);

    if (threadIdx.x == 0)
        atomicAdd(c, sum / (float)n);
}

void sqrt_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_sqrt<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void dyntanh_cuda(const float* x, float a, float b, float g, float* y, int64_t n, ag_cuda_stream_t s) {
  k_dyntanh<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, a, b, g, y, n);
}
void mseloss_cuda(const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s) {
  k_mseloss<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(a, b, c, n);
}
void add_cuda(const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s) {
  k_add<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(a, b, c, n);
}
void sub_cuda(const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s) {
  k_sub<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(a, b, c, n);
}
void hadmul_cuda(const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s) {
  k_mul<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(a, b, c, n);
}
void div_cuda(const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s) {
  k_div<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(a, b, c, n);
}
void pow_cuda(const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s) {
  k_pow<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(a, b, c, n);
}
void square_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_square<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void neg_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_neg<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void clip_cuda(const float* x, float* y, float minv, float maxv, int64_t n, ag_cuda_stream_t s) {
  k_clip<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, minv, maxv, n);
}


// Activations
void sign_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_sign<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void cos_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_cos<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void sin_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_sin<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void cosh_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_cosh<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void sinh_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_sinh<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void relu_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_relu<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void relumask_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_relu<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void leaky_relu_cuda(const float* x, float* y, float slope, int64_t n, ag_cuda_stream_t s) {
  k_leaky_relu<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, slope, n);
}
void gelu_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_gelu<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void silu_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_silu<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void mish_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_mish<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void tanh_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_tanh<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void sigmoid_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_sigmoid<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void hard_sigmoid_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_hard_sigmoid<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void hard_swish_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_hard_swish<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void exp_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_exp<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void log_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_log<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void softplus_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_softplus<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}

void gcu_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_gcu<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void lisht_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_lisht<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void gauss_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_gauss<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void parcon_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_parcon<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void reci_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_reci<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}

__global__ void k_maeloss(const float* __restrict__ p, const float* __restrict__ t,
                      float* __restrict__ c, int64_t n) {
    float sum = 0.0f;

    // grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x)
    {
        float diff = p[i] - t[i];
        sum += fabs(diff);
    }

    // block reduce
    sum = blockReduceSum(sum);

    if (threadIdx.x == 0)
        atomicAdd(c, sum / (float)n);
}


void maeloss_cuda(const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s) {
  k_maeloss<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(a, b, c, n);
}