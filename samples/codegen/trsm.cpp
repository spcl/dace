#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
// solves AX = B, for A being upper triangular matrix
template <typename T>
void trsm_cpu(const T* A, const T* B, T* X, int n) {
  // backward substitituon
  for (int i = n - 1; i >= 0; i--) {
    T sum = 0;
    for (int j = i + 1; j < n; j++) {
      sum += A[i * n + j] * X[j];
    }
    X[i] = (B[i] - sum) / A[i * n + i];
  }
}

// CPU implementation of TRSM (row-major order)
void trsm_cpu_claude(const std::vector<float>& A, std::vector<float>& B, int n,
                     int m) {
  for (int k = n - 1; k >= 0; k--) {
    for (int j = 0; j < m; j++) {
      B[k * m + j] /= A[k * n + k];
      for (int i = 0; i < k; i++) {
        B[i * m + j] -= A[i * n + k] * B[k * m + j];
      }
    }
  }
}

// CPU implementation of TRSM (row-major order)
void trsm_cpu_claude_vector(const std::vector<float>& A, std::vector<float>& B,
                            int n) {
  for (int k = n - 1; k >= 0; k--) {
    for (int i = 0; i < k; i++) {
      B[i] -= A[i * n + k] * B[k];
    }
    B[k] /= A[k * n + k];
  }
}

// transpose a matrix
template <typename T>
void transpose(const T* A, T* At, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      At[j * n + i] = A[i * n + j];
    }
  }
}

template <typename T>
bool compare_results(const T* A_1, const T* A_2, int n,
                     double tolerance = 1e-5) {
  for (int i = 0; i < n; i++) {
    // std::cout << "\n" << A_1[i] << " " << A_2[i];
    // std::flush(std::cout);
    if (std::abs(A_1[i] - A_2[i]) > tolerance) {
      return false;
    }
  }
  return true;
}

// GPU implementation using cuBLAS
void trsm_gpu(const float* A, float* B, int n, int m) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  cublasSideMode_t side = CUBLAS_SIDE_LEFT;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  cublasOperation_t trans = CUBLAS_OP_N;
  cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

  // Note: cuBLAS uses column-major order, so we swap n and m
  cublasStrsm(handle, side, uplo, trans, diag, n, m, &alpha, A, n, B, n);

  cublasDestroy(handle);
}

int main() {
  const int n = 4;  // Size of the square matrix A

  // Initialize matrix A (upper triangular)
  std::vector<float> A = {2, 1, -1, 0.5, 0, 1, 3, 1, 0, 0, 1, 1, 0, 0, 0, 1};

  // initialize vector B
  std::vector<float> B = {1, 0, 5, 1};

  // initialize vector X
  std::vector<float> X = {0, 0, 0, 0};

  // print matrix A:
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << A[i * n + j] << " ";
    }
    std::cout << "\n";
  }

  // print vector B:
  std::cout << "\nB\n";
  for (int i = 0; i < n; i++) {
    std::cout << B[i] << "\n";
  }

  // Perform CPU TRSM
  trsm_cpu(&A[0], &B[0], &X[0], n);

  //   trsm_cpu_claude_vector(A, B, n);

  // allocate GPU memory
  float *d_A, *d_B;
  float* h_B;
  h_B = (float*)malloc(n * sizeof(float));
  cudaMalloc(&d_A, n * n * sizeof(float));
  cudaMalloc(&d_B, n * sizeof(float));

  float* A_transposed = (float*)malloc(n * n * sizeof(float));
  transpose(&A[0], A_transposed, n);

  // copy data to GPU
  cudaMemcpy(d_A, &A_transposed[0], n * n * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, &B[0], n * sizeof(float), cudaMemcpyHostToDevice);

  // cuBLAS
  //   trsm_gpu(d_A, d_B, n, 1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
              CUBLAS_DIAG_NON_UNIT, n, 1, &alpha, d_A, n, d_B, n);

  cublasDestroy(handle);
  // copy result back to host
  cudaMemcpy(h_B, d_B, n * sizeof(float), cudaMemcpyDeviceToHost);

  //   std::cout << "\n\nResult:\n";
  //   // print the result
  //   for (int i = 0; i < n; i++) {
  //     std::cout << X[i] << " ";
  //   }

  //   // print the result
  //   std::cout << "\nB\n";
  //   for (int i = 0; i < n; i++) {
  //     std::cout << B[i] << " ";
  //   }

  std::cout << "\nAbout to compare results. manual:\n";
  for (int i = 0; i < n; i++) {
    std::cout << "\n" << X[i];
  }
  std::cout << "\nCublas:\n";
  for (int i = 0; i < n; i++) {
    std::cout << "\n" << h_B[i];
  }

  bool correct = compare_results(&X[0], &h_B[0], n);
  std::cout << "\n\nResults " << (correct ? "match" : "do not match") << "\n";

  cudaFree(d_A);
  cudaFree(d_B);
}