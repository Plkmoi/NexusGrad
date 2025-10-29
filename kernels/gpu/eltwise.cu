// kernels/gpu/eltwise.cu
#include <cuda_runtime.h>
#include <cstdint>
#include "ad/kernels_api.hpp"
#include <math_constants.h>


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