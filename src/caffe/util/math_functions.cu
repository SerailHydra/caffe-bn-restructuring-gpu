#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>
#include "caffe/common.hpp"
#include <iostream>
#include <cmath>

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/sgemm_traits.h"
#include <vector>
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

typedef cutlass::gemm::Gemm<cutlass::gemm::SgemmTraits<
cutlass::MatrixLayout::kColumnMajor,
  cutlass::MatrixLayout::kColumnMajor,
  cutlass::Shape<8, 128, 128>>> Gemm_nn;

typedef cutlass::gemm::Gemm<cutlass::gemm::SgemmTraits<
cutlass::MatrixLayout::kRowMajor,
  cutlass::MatrixLayout::kColumnMajor,
  cutlass::Shape<8, 128, 128>>> Gemm_tn;

typedef cutlass::gemm::Gemm<cutlass::gemm::SgemmTraits<
cutlass::MatrixLayout::kColumnMajor,
  cutlass::MatrixLayout::kRowMajor,
  cutlass::Shape<8, 128, 128>>> Gemm_nt;

typedef cutlass::gemm::Gemm<cutlass::gemm::SgemmTraits<
cutlass::MatrixLayout::kRowMajor,
  cutlass::MatrixLayout::kRowMajor,
  cutlass::Shape<8, 128, 128>>> Gemm_tt;

// norm fusion
typedef cutlass::gemm::Gemm_norm<cutlass::gemm::SgemmTraits<
cutlass::MatrixLayout::kColumnMajor,
  cutlass::MatrixLayout::kColumnMajor,
  cutlass::Shape<8, 128, 128>>> Gemm_nn_norm;

// mean-var fusion
typedef cutlass::gemm::Gemm_gather2<cutlass::gemm::SgemmTraits<
cutlass::MatrixLayout::kColumnMajor,
  cutlass::MatrixLayout::kColumnMajor,
  cutlass::Shape<8, 128, 128>>> Gemm_nn_gather;

// mean-var-norm fusion
typedef cutlass::gemm::Gemm_gather_norm<cutlass::gemm::SgemmTraits<
cutlass::MatrixLayout::kColumnMajor,
  cutlass::MatrixLayout::kColumnMajor,
  cutlass::Shape<8, 128, 128>>> Gemm_nn_gather_norm;

typedef cutlass::gemm::Gemm_relu<cutlass::gemm::SgemmTraits<
cutlass::MatrixLayout::kRowMajor,
  cutlass::MatrixLayout::kColumnMajor,
  cutlass::Shape<8, 128, 128>>> Gemm_tn_relu;

typedef cutlass::gemm::Gemm_relu<cutlass::gemm::SgemmTraits<
cutlass::MatrixLayout::kColumnMajor,
  cutlass::MatrixLayout::kRowMajor,
  cutlass::Shape<8, 128, 128>>> Gemm_nt_relu;

namespace caffe {

// not implemented
template <>
void caffe_gpu_gemm_norm <double>(CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int, double, double const*, double const*, double, double*, double const*, double const*, double const*, double const*, double *){};
template <>
void caffe_gpu_gemm_mean_var_fusion <double>(CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int, double, double const*, double const*, double, double*, double*, double*){};
template <>
void caffe_gpu_gemm_relu<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C){};
template <>
void caffe_gpu_gemm_mean_var_norm_fusion <double>(CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int, double, double const*, double const*, double, double*, double*, double*, double const*, double const*, double const*, double const*, double *){};

template <>
void caffe_gpu_gemm_mean_var_fusion<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C, float* d_x_temp, float* d_x2_temp) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  if (TransB == CblasNoTrans && TransA == CblasNoTrans) {
    Gemm_nn_gather::Params params;
    params.initialize(N, M, K, alpha, B, ldb, A, lda, beta, C, ldc, C, ldc, d_x_temp, d_x2_temp);
    Gemm_nn_gather::launch(params);
  }
  else {
    LOG(FATAL) << "No mean-var fusion support for Transposed matrix";
  }
}

template <>
void caffe_gpu_gemm_mean_var_norm_fusion<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C, float* d_x_temp, float* d_x2_temp, const float* mean, const float* var, const float* gamma, const float* beta_norm, float* relu_ptr) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  if (TransB == CblasNoTrans && TransA == CblasNoTrans) {
    Gemm_nn_gather_norm::Params params;
    params.initialize(N, M, K, alpha, B, ldb, A, lda, beta, C, ldc, C, ldc, d_x_temp, d_x2_temp, const_cast<float *> (mean), const_cast<float *> (var), const_cast<float *> (gamma), const_cast<float *> (beta_norm), static_cast<float *>(relu_ptr));
    Gemm_nn_gather_norm::launch(params);
  }
  else {
    LOG(FATAL) << "No mean-var fusion support for Transposed matrix";
  }
}

template <>
void caffe_gpu_gemm_norm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C,
    const float* mean,
    const float* var,
    const float* gamma,
    const float* beta_norm,
    float* relu_ptr) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;

  if (TransB == CblasNoTrans && TransA == CblasNoTrans) {
    Gemm_nn_norm::Params params;
    params.initialize(N, M, K, alpha, B, ldb, A, lda, beta, C, ldc, C, ldc, const_cast<float *> (mean), const_cast<float *> (var), const_cast<float *> (gamma), const_cast<float *> (beta_norm), static_cast<float *> (relu_ptr));
    Gemm_nn_norm::launch(params);
   }
  else {
    LOG(FATAL) << "No norm fusion supoprt for Transposed matrix";
  }
}

template <typename Dtype>
__global__ void norm_save_relu_kernel(const int ch, const int spatial, Dtype* input, const Dtype* mean, const Dtype* var, const Dtype *gamma, const Dtype *beta, Dtype* relu_inp) {
  CUDA_KERNEL_LOOP(index, ch * spatial) {
    int channel_idx = index / spatial;
    relu_inp[index] = gamma[channel_idx] * input[index]  + beta[channel_idx];
    input[index] = relu_inp[index] > 0? relu_inp[index] : 0;
  }
}

template <typename Dtype>
__global__ void norm_save_relu_kernel(const int ch, const int spatial, Dtype* input, const Dtype* mean, const Dtype* var, const Dtype *gamma, const Dtype *beta, Dtype* relu_inp, Dtype* dst) {
  CUDA_KERNEL_LOOP(index, ch * spatial) {
    int channel_idx = index / spatial;
    relu_inp[index] = gamma[channel_idx] * input[index]  + beta[channel_idx];
    dst[index] = relu_inp[index] > 0? relu_inp[index] : 0;
  }
}

template <>
void norm_and_save_relu<float>(const int ch, const int spatial, float* input, const float* mean, const float* var, const float *gamma, const float *beta, float* relu_inp, float* dst)
{
  if (dst == NULL) {
    norm_save_relu_kernel<float> <<< CAFFE_GET_BLOCKS(ch * spatial), CAFFE_CUDA_NUM_THREADS>>> (
        ch, spatial, input, mean, var, gamma, beta, relu_inp);
  }
  else {
  norm_save_relu_kernel<float> <<< CAFFE_GET_BLOCKS(ch * spatial), CAFFE_CUDA_NUM_THREADS>>> (
      ch, spatial, input, mean, var, gamma, beta, relu_inp, dst);
  }
}

template <>
void norm_and_save_relu<double>(const int ch, const int spatial, double* input, const double* mean, const double* var, const double *gamma, const double *beta, double* relu_inp, double *dst)
{
}

template <typename Dtype>
__global__ void relu_ker(const int n, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    if (y[index] < 0) {
      y[index] = 0;
    }
  }
}

template <>
void ReLU_inplace<float> (const int n, float* x) {
  relu_ker<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x);
}

template <>
void ReLU_inplace<double> (const int n, double* x) {
}

template <>
void caffe_gpu_gemm_relu<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;

  if (TransA == CblasNoTrans && TransB == CblasTrans) {
    Gemm_tn_relu::Params params;
    params.initialize(N, M, K, alpha, B, ldb, A, lda, beta, C, ldc, C, ldc);
    Gemm_tn_relu::launch(params);
  }
  else if (TransA == CblasTrans && TransB == CblasNoTrans) {
    Gemm_nt_relu::Params params;
    params.initialize(N, M, K, alpha, B, ldb, A, lda, beta, C, ldc, C, ldc);
    Gemm_nt_relu::launch(params);
  }
  else {
    LOG(FATAL) << "No support other than NT or TN type in 1x1 backward fusion";
  }
}

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  if (TransB == CblasNoTrans) {
    if (TransA == CblasNoTrans) {
      Gemm_nn::Params params;
      params.initialize(N, M, K, alpha, B, ldb, A, lda, beta, C, ldc, C, ldc);
      Gemm_nn::launch(params);
    }
    else {
      Gemm_nt::Params params;
      params.initialize(N, M, K, alpha, B, ldb, A, lda, beta, C, ldc, C, ldc);
      Gemm_nt::launch(params);
    }
  }
  else if (TransB != CblasNoTrans) {
    if (TransA == CblasNoTrans) {
      Gemm_tn::Params params;
      params.initialize(N, M, K, alpha, B, ldb, A, lda, beta, C, ldc, C, ldc);
      Gemm_tn::launch(params);
    }
    else {
      Gemm_tt::Params params;
      params.initialize(N, M, K, alpha, B, ldb, A, lda, beta, C, ldc, C, ldc);
      Gemm_tt::launch(params);
    }
  }
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void relu_gather_kernel(const int ch, const int spatial, const Dtype* x_relu, const Dtype* x_bn, Dtype* input, Dtype* d_gamma, Dtype* d_beta, const Dtype* mean, const Dtype* var) {
  // for every pixel of CHW
  CUDA_KERNEL_LOOP(index, ch * spatial) {
    int ch = index / spatial;
    if (x_relu[index] > 0) {
      d_beta[index] += input[index];
      d_gamma[index] += input[index] * (x_bn[index] - mean[ch]) / var[ch];
    }
    else {
      input[index] = 0;
    }
  }
}

template <>
void caffe_gpu_relu_gather<float> (const int ch, const int spatial, const float* x_relu, const float* x_bn, const float* input, float* d_gamma, float* d_beta, const float* mean, const float* var) {
  relu_gather_kernel<float> <<<CAFFE_GET_BLOCKS(ch * spatial), CAFFE_CUDA_NUM_THREADS>>> (ch, spatial, x_relu, x_bn, const_cast<float *> (input), d_gamma, d_beta, mean, var);
}

template <>
void caffe_gpu_relu_gather<double> (const int ch, const int spatial, const double* x_relu, const double* x_bn, const double* input, double* d_gamma, double* d_beta, const double* mean, const double*var) {
}

template <typename Dtype>
__global__ void ABC_kernel(const int ch, const int N, const Dtype* gamma, const Dtype* var, const Dtype* d_gamma, const Dtype* d_beta, const Dtype* mean, Dtype* A, Dtype* B, Dtype* C) {
  CUDA_KERNEL_LOOP(index, ch) {
    A[index] = gamma[index] / var[index];
    B[index] = - gamma[index] * d_gamma[index] / (N * var[index] * var[index]);
    C[index] = gamma[index] / (N * var[index]) * (mean[index] * d_gamma[index] / var[index] - d_beta[index]);
  }
}

template <>
void caffe_gpu_compute_ABC<float>(const int ch, const int N, const float* gamma, const float* var, const float* d_gamma, const float* d_beta, const float* mean, float* A, float* B, float* C) {
  ABC_kernel<float> <<<CAFFE_GET_BLOCKS(ch), CAFFE_CUDA_NUM_THREADS>>> (
      ch, N, gamma, var, d_gamma, d_beta, mean, A, B, C);
}

template <>
void caffe_gpu_compute_ABC<double>(const int ch, const int N, const double* gamma, const double* var, const double* d_gamma, const double* d_beta, const double* mean, double* A, double* B, double* C) {}

template <typename Dtype>
__global__ void transform_kernel(const int ch, const int spatial, const Dtype* A, const Dtype* B, const Dtype* C, Dtype* top_diff, const Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, ch * spatial) {
    int ch = index / spatial;
    top_diff[index] = A[ch] * top_diff[index] + B[ch] * top_data[index] + C[ch];
  }
}

template <>
void caffe_gpu_top_diff_transform<float>(const int ch, const int spatial, const float* A, const float* B, const float* C, float* top_diff, const float* top_data) {
  transform_kernel<float> <<<CAFFE_GET_BLOCKS(ch * spatial), CAFFE_CUDA_NUM_THREADS>>> (ch, spatial, A, B, C, top_diff, top_data);
}

template <>
void caffe_gpu_top_diff_transform<double>(const int ch, const int spatial, const double* A, const double* B, const double* C, double* top_diff, const double* top_data) {
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
