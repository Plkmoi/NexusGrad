// ====================================================================
// FILE: kernels/gpu/vjp.cu (The Final, Correct Version)
// ====================================================================
#include <cuda_runtime.h>
#include <cstdint>
#include "ad/kernels_api.hpp"
#include <cfloat>

__device__ __forceinline__ float sigmoidf(float x) {
  return 1.f / (1.f + __expf(-x));
}

__global__ void k_vjp_add_accum(float* gA, float* gB, const float* gy, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&gA[i], gy[i]);
        atomicAdd(&gB[i], gy[i]);
    }
}

void vjp_add_cuda(float* gA, float* gB, const float* gy,
                  int64_t n, ag_cuda_stream_t s) {
    dim3 blocks( (unsigned int)((n + 255) / 256) );
    k_vjp_add_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gA, gB, gy, n);
}


__global__ void k_vjp_relu_accum(float* gX, const float* gy, const float* X, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Only add the gradient if the original input (X) was positive.
        if (X[i] > 0.0f) {
            atomicAdd(&gX[i], gy[i]);
        }
    }
}

void vjp_relu_cuda(float* gX, const float* gy, const float* X, int64_t n, ag_cuda_stream_t s) {
    dim3 blocks( (unsigned int)((n + 255) / 256) );
    k_vjp_relu_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, gy, X, n);
}

// __global__ void k_vjp_leaky_relu_accum(float* gX, const float* gy, const float* X, const float * slope, int64_t n) {
//     int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) {
//         // Only add the gradient if the original input (X) was positive.
//         if (X[i] > 0.0f) {
//             atomicAdd(&gX[i], gy[i]);
//         }
//         else{
// atomicAdd(&gX[i], gy[i]*slope[0]);

//         }
//     }
// }

// void vjp_leaky_relu_cuda(float* gX, const float* gy, const float* X, const float * slope, int64_t n, ag_cuda_stream_t s) {
//     dim3 blocks( (unsigned int)((n + 255) / 256) );
//     k_vjp_leaky_relu_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, gy, X, slope, n);
// }



__global__ void k_vjp_tanh_accum(float* gX, const float* gy, const float* X, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Only add the gradient if the original input (X) was positive.
            atomicAdd(&gX[i], (gy[i] *(1- (X[i]*X[i]))   ));
    }
}

void vjp_tanh_cuda(float* gX, const float* gy, const float* X, int64_t n, ag_cuda_stream_t s) {
    dim3 blocks( (unsigned int)((n + 255) / 256) );
    k_vjp_tanh_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, gy, X, n);
}

__global__ void k_vjp_sub_accum(float* gA, float* gB, const float* gy, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&gA[i], gy[i]);
        atomicAdd(&gB[i], -gy[i]);
    }
}

void vjp_sub_cuda(float* gA, float* gB, const float* gy,
                  int64_t n, ag_cuda_stream_t s) {
    dim3 blocks( (unsigned int)((n + 255) / 256) );
    k_vjp_sub_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gA, gB, gy, n);
}

__global__ void k_vjp_hadmul_accum(float* gA, float* gB, const float* A, const float* B, const float* gy, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&gA[i], gy[i]*B[i]);
        atomicAdd(&gB[i], gy[i]*A[i]);
    }
}

void vjp_hadmul_cuda(float* gA, float* gB, const float* gy, const float* A, const float* B, 
                  int64_t n, ag_cuda_stream_t s) {
    dim3 blocks( (unsigned int)((n + 255) / 256) );
    k_vjp_hadmul_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gA, gB, A, B, gy, n);
}

__global__ void k_vjp_div_accum(float* gA, float* gB, const float* A, const float* B, const float* gy, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&gA[i], gy[i]/B[i]);
        atomicAdd(&gB[i], -A[i]*gy[i]/(B[i]*B[i]));
    }
}

void vjp_div_cuda(float* gA, float* gB, const float* gy, const float* A, const float* B, 
                  int64_t n, ag_cuda_stream_t s) {
    dim3 blocks( (unsigned int)((n + 255) / 256) );
    k_vjp_div_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gA, gB, A, B, gy, n);
}



__global__ void k_vjp_pow_accum(float* gA, float* gB, const float* __restrict__ A,
                                const float* __restrict__ B, const float* __restrict__ gy, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float a = A[i];
    float b = B[i];
    float g = gy[i];

    // p = a^b (note: if a <= 0, pow can be problematic for non-integer b)
    float p = __powf(a, b);

    // d/dA: b * a^(b-1)  (if a == 0 and b < 1 this is inf; assume typical positive inputs)
    if (a != 0.0f) {
      atomicAdd(&gA[i], g * b * __powf(a, b - 1.0f));
    } else {
      // when a==0, treat derivative carefully: if b > 1 -> 0; otherwise leave 0
      if (b > 1.0f) atomicAdd(&gA[i], 0.0f);
    }

    // d/dB: a^b * log(a)  (only valid for a > 0)
    if (a > 0.0f) {
      atomicAdd(&gB[i], g * p * __logf(a));
    } else {
      // if a <= 0, avoid log; set derivative to 0 (common practical fallback)
      atomicAdd(&gB[i], 0.0f);
    }
  }
}
void vjp_pow_cuda(float* gA, float* gB, const float* A, const float* B, const float* gy, int64_t n, ag_cuda_stream_t s) {
  dim3 blocks((unsigned int)((n + 255) / 256));
  k_vjp_pow_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gA, gB, A, B, gy, n);

}

__global__ void k_vjp_square_accum(float* gX, const float* __restrict__ x, const float* __restrict__ gy, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&gX[i], gy[i] * (2.0f * x[i]));
  }
}
void vjp_square_cuda(float* gX, const float* x, const float* gy, int64_t n, ag_cuda_stream_t s) {
    dim3 blocks((unsigned int)((n + 255) / 256));
  k_vjp_square_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, x, gy, n);
}

__global__ void k_vjp_neg_accum(float* gX, const float* __restrict__ gy, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&gX[i], -gy[i]);
  }
}
void vjp_neg_cuda(float* gX, const float* gy, int64_t n, ag_cuda_stream_t s) {
    dim3 blocks((unsigned int)((n + 255) / 256));
  k_vjp_neg_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, gy, n);
}

__global__ void k_vjp_clip_accum(float* gX, const float* __restrict__ x, float minv, float maxv,
                                 const float* __restrict__ gy, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float xi = x[i];
    if (xi > minv && xi < maxv) {
      atomicAdd(&gX[i], gy[i]);
    } else {
      // outside clip range -> gradient is 0
    }
  }
}
void vjp_clip_cuda(float* gX, const float* x, float minv, float maxv, const float* gy, int64_t n, ag_cuda_stream_t s) {
  int64_t blocks((unsigned int)((n + 255) / 256));
    k_vjp_clip_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, x, minv, maxv, gy, n);
}

// --------------------
// Activations VJPs
// --------------------


__global__ void k_vjp_leaky_relu_accum(float* gX, const float* __restrict__ X,
                                       const float* __restrict__ gy, float slope, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float xi = X[i];
    atomicAdd(&gX[i], gy[i] * (xi > 0.f ? 1.0f : slope));
  }
}
void vjp_leaky_relu_cuda(float* gX, const float* X, const float* gy, float slope, int64_t n, ag_cuda_stream_t s) {
    int64_t blocks((unsigned int)((n + 255) / 256));
    k_vjp_leaky_relu_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, X, gy, slope, n);
}

// sigmoid, tanh, silu, gelu, mish, exp

__global__ void k_vjp_sigmoid_accum(float* gX, const float* __restrict__ X, const float* __restrict__ gy, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float s = 1.0f / (1.0f + __expf(-X[i])); // sigmoid(X)
    atomicAdd(&gX[i], gy[i] * s * (1.0f - s));
  }
}
void vjp_sigmoid_cuda(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s) {
    int64_t blocks((unsigned int)((n + 255) / 256));
  k_vjp_sigmoid_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, X, gy, n);
}

__global__ void k_vjp_silu_accum(float* gX, const float* __restrict__ X, const float* __restrict__ gy, int64_t n) {
  // SiLU(x) = x * sigmoid(x)
  // d/dx = sigmoid(x) + x * sigmoid(x)*(1-sigmoid(x))
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float xi = X[i];
    float s = 1.0f / (1.0f + __expf(-xi));
    atomicAdd(&gX[i], gy[i] * (s + xi * s * (1.0f - s)));
  }
}
void vjp_silu_cuda(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s) {
    int64_t blocks((unsigned int)((n + 255) / 256));
  k_vjp_silu_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, X, gy, n);
}

// GELU derivative (approximate) using same constants as forward
__global__ void k_vjp_gelu_accum(float* gX, const float* __restrict__ X, const float* __restrict__ gy, int64_t n) {
  const float k0 = 0.7978845608f; // sqrt(2/pi)
  const float k1 = 0.044715f;
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = X[i];
    float xx3 = x * x * x;
    float u = k0 * (x + k1 * xx3);
    float t = tanhf(u);
    float phi = 0.5f * (1.0f + t);
    // derivative phi' = 0.5 * (1 - t^2) * du/dx
    float du_dx = k0 * (1.0f + 3.0f * k1 * x * x);
    float dphi = 0.5f * (1.0f - t * t) * du_dx;
    // d gelu = phi + x * dphi
    float dg = phi + x * dphi;
    atomicAdd(&gX[i], gy[i] * dg);
  }
}
void vjp_gelu_cuda(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s) {
    int64_t blocks((unsigned int)((n + 255) / 256));
  k_vjp_gelu_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, X, gy, n);
}

// MISH derivative (using softplus = log(1+exp(x)) and its derivatives)
// d/dx [x * tanh(softplus(x))] = tanh(sp) + x * sech^2(sp) * sigmoid(x)
__global__ void k_vjp_mish_accum(float* gX, const float* __restrict__ X, const float* __restrict__ gy, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = X[i];
    float sp = __logf(1.0f + __expf(x));         // softplus
    float t = tanhf(sp);                        // tanh(softplus)
    float s = 1.0f / (1.0f + __expf(-x));      // sigmoid(x)
    float sech2 = 1.0f - t * t;                // sech^2(sp)
    float grad = t + x * sech2 * s;
    atomicAdd(&gX[i], gy[i] * grad);
  }
}
void vjp_mish_cuda(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s) {
    int64_t blocks((unsigned int)((n + 255) / 256));
  k_vjp_mish_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, X, gy, n);
}

__global__ void k_vjp_exp_accum(float* gX, const float* __restrict__ X, const float* __restrict__ gy, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = __expf(X[i]);
    atomicAdd(&gX[i], gy[i] * v);
  }
}
void vjp_exp_cuda(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s) {
    int64_t blocks((unsigned int)((n + 255) / 256));
  k_vjp_exp_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, X, gy, n);
}

// Hard sigmoid: forward v = 0.2*x + 0.5; clipped to [0,1].
// derivative is 0.2 when 0<v<1 else 0.
__global__ void k_vjp_hard_sigmoid_accum(float* gX, const float* __restrict__ X, const float* __restrict__ gy, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = 0.2f * X[i] + 0.5f;
    if (v > 0.0f && v < 1.0f) atomicAdd(&gX[i], gy[i] * 0.2f);
  }
}
void vjp_hard_sigmoid_cuda(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s) {
    int64_t blocks((unsigned int)((n + 255) / 256));
  k_vjp_hard_sigmoid_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, X, gy, n);
}

// Hard swish: x * hard_sigmoid(x) ; derivative = hard_sigmoid(x) + x * hard_sigmoid'(x)
// hard_sigmoid'(x) = 0.2 when inside (0,1) else 0.
__global__ void k_vjp_hard_swish_accum(float* gX, const float* __restrict__ X, const float* __restrict__ gy, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = X[i];
    float hs = 0.2f * x + 0.5f;
    float h = (hs > 1.0f) ? 1.0f : (hs < 0.0f ? 0.0f : hs);
    float hprime = (hs > 0.0f && hs < 1.0f) ? 0.2f : 0.0f;
    float grad = h + x * hprime;
    atomicAdd(&gX[i], gy[i] * grad);
  }
}
void vjp_hard_swish_cuda(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s) {
    int64_t blocks((unsigned int)((n + 255) / 256));
  k_vjp_hard_swish_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, X, gy, n);
}

__global__ void k_vjp_sofba_accum(float* gX, const float* __restrict__ X, const float* __restrict__ gy, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&gX[i], gy[i] * (1.0f / (1.0f + __expf(-X[i]))));
  }
}
void vjp_sofba_cuda(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s) {
    int64_t blocks((unsigned int)((n + 255) / 256));
  k_vjp_sofba_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, X, gy, n);
}


__global__ void k_vjp_log_accum(float* gX, const float* __restrict__ X, const float* __restrict__ gy, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && X[i] != 0.0f) {
    atomicAdd(&gX[i], gy[i] /X[i]);
  }
  else{

    atomicAdd(&gX[i], 0.0f);
  }
}
void vjp_log_cuda(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s) {
    int64_t blocks((unsigned int)((n + 255) / 256));
  k_vjp_log_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, X, gy, n);
}


__global__ void k_vjp_mseloss_accum(float* gA, float* gB, const float* A, const float* B, const float* gy, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&gA[i], gy[i]*(A[i]-B[i])*2.f/n);
        atomicAdd(&gB[i], gy[i]*(B[i]-A[i])*2.f/n);
    }
}

void vjp_mseloss_cuda(float* gA, float* gB, const float* gy, const float* A, const float* B, 
                  int64_t n, ag_cuda_stream_t s) {
    dim3 blocks( (unsigned int)((n + 255) / 256) );
    k_vjp_mseloss_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gA, gB, A, B, gy, n);
}

__global__ void k_vjp_sum(float* gX, const float* __restrict__ X, const float* __restrict__ gy, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {

    atomicAdd(&gX[i], gy[i]);
  }
}
void vjp_sum_cuda(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s) {
    int64_t blocks((unsigned int)((n + 255) / 256));
  k_vjp_sum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, X, gy, n);
}


__global__ void k_vjp_maeloss_accum(float* gA, float* gB, const float* A, const float* B, const float* gy, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&gA[i], gy[i]* ( (A[i]-B[i]) >0 ? 1.f :((A[i]-B[i]) <0 )?-1.f:0.f)  *1.f/n);
        atomicAdd(&gB[i], gy[i]*  ( (B[i]-A[i]) >0 ? 1.f :((B[i]-A[i]) <0 )?-1.f:0.f) *1.f/n);
    }
}

void vjp_maeloss_cuda(float* gA, float* gB, const float* gy, const float* A, const float* B, 
                  int64_t n, ag_cuda_stream_t s) {
    dim3 blocks( (unsigned int)((n + 255) / 256) );
    k_vjp_maeloss_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gA, gB, A, B, gy, n);
}


__global__ void k_vjp_rowmax(float* gX, 
                              const float* __restrict__ X, 
                              const float* __restrict__ Y,  // Forward pass output (rowmax values)
                              const float* __restrict__ gy, 
                              int rows, int cols) {
    // Each thread handles one row
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    // Load the max value and gradient for this row
    float row_max = Y[row];
    float grad = gy[row];

    // Loop over all columns of this row
    for (int col = 0; col < cols; col++) {
        int idx = row * cols + col;

        // If the input equals the max value, assign grad; else 0
        gX[idx] = (X[idx] == row_max) ? gy[row] : 0.0f;
    }
}

void vjp_rowmax_cuda(float* gX, const float* X, const float* Y, const float* gy, int64_t n, int64_t w, ag_cuda_stream_t s) {
    dim3 blocks( (unsigned int)((n + 255) / 256) );
  k_vjp_rowmax<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, X, Y, gy, n, w);
}


__global__ void k_vjp_rowsum(float* gX, 
                              const float* __restrict__ X, 
                              const float* __restrict__ Y,  // Forward pass output (rowmax values)
                              const float* __restrict__ gy, 
                              int rows, int cols) {
    // Each thread handles one row
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    // Load the max value and gradient for this row
    float row_max = Y[row];
    float grad = gy[row];

    // Loop over all columns of this row
    for (int col = 0; col < cols; col++) {
        int idx = row * cols + col;

        // If the input equals the max value, assign grad; else 0
        gX[idx] = gy[row];
    }
}

void vjp_rowsum_cuda(float* gX, const float* X, const float* Y, const float* gy, int64_t n, int64_t w, ag_cuda_stream_t s) {
    dim3 blocks( (unsigned int)((n + 255) / 256) );
  k_vjp_rowsum<<<blocks, 256, 0, (cudaStream_t)s>>>(gX, X, Y, gy, n, w);
}

