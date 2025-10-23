#pragma once
void run_cuda_matrix(const float* A, const float* B, float* C, int M, int K, int N); // VALSEND
void run_cuda_gemm(const float* A, const float* B,  const float* C, float* E, int M, int K, int N); // VALSEND
void run_cuda_relu(const float* A, float* B, int N); // VALSEND
void run_cuda_relumask(const float* A, float* B, int N);  // VALSEND
void run_cuda_exp(const float* A, float* B, int N);  // VALSEND
void run_cuda_sigmoid(const float* A, float* B, int N); // VALSEND
void run_cuda_sigmoidiff(const float* A, float* B, int N); // VALSEND
void run_cuda_add(const float* A, const float* B, float* C, int width);  // VALSEND
void run_cuda_sub(const float* A, const float* B, float* C, int width); // VALSEND
void run_cuda_hadmul(const float* A, const float* B, float* C, int width); // VALSEND
void run_flash_forward(const float* Q, const float* K, const float* V, float* O, int B, int nh, int N, int d);  // old legacy
void run_cuda_div(const float* A, const float* B, float* C, int width); // VALSEND
void run_cuda_gcu(const float*, float*, int); // VALSEND
void run_cuda_mish(const float*, float*, int); // VALSEND
void run_cuda_gaus(const float*, float*, int); // VALSEND
void run_cuda_parcon(const float*, float*, int); // VALSEND
void run_cuda_lisht(const float*, float*, int); // VALSEND
void run_cuda_softplus(const float*, float*, int); // VALSEND
void run_cuda_silu(const float*, float*, int); // VALSEND
void run_cuda_gelu(const float*, float*, int); // VALSEND
void run_cuda_log(const float*, float*, int); // VALSEND
void run_cuda_tanh(const float*, float*, int); // VALSEND
void run_cuda_rowsum(const float* X, float* Y, int rows, int cols); // VALSEND
void run_cuda_rowmax(const float* X, float* Y, int rows, int cols); // VALSEND
void run_cuda_leakyrelu(const float* A, const float* H, float* B, int N); // VALSEND
void run_flash_forwarde(const float* Q, const float* K, const float* V, float* O, int B, int nh, int N, int d); // VALSEND

