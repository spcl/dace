#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

// CPU implementation of TRSM (row-major order)
void trsm_cpu(const std::vector<float>& A, std::vector<float>& B, int n,
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

// Function to compare CPU and GPU results
bool compare_results(const std::vector<float>& cpu_result,
                     const std::vector<float>& gpu_result,
                     float tolerance = 1e-5) {
  if (cpu_result.size() != gpu_result.size()) {
    return false;
  }

  for (size_t i = 0; i < cpu_result.size(); ++i) {
    if (std::abs(cpu_result[i] - gpu_result[i]) > tolerance) {
      return false;
    }
  }

  return true;
}

// Function to transpose a matrix
std::vector<float> transpose(const std::vector<float>& matrix, int rows,
                             int cols) {
  std::vector<float> transposed(rows * cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      transposed[j * rows + i] = matrix[i * cols + j];
    }
  }
  return transposed;
}

int main() {
  const int n = 4;  // Size of the square matrix A
  const int m = 2;  // Number of columns in matrix B

  // Initialize matrix A (upper triangular)
  std::vector<float> A = {2.0f, 1.0f, 3.0f, 2.0f, 0.0f, 4.0f, 1.0f, 5.0f,
                          0.0f, 0.0f, 3.0f, 2.0f, 0.0f, 0.0f, 0.0f, 1.0f};

  // Initialize matrix B
  std::vector<float> B = {8.0f, 7.0f, 20.0f, 14.0f, 13.0f, 8.0f, 4.0f, 2.0f};

  // Create copies for GPU computation
  std::vector<float> B_gpu = B;

  // Perform CPU TRSM
  trsm_cpu(A, B, n, m);

  // Transpose A and B for cuBLAS (column-major order)
  std::vector<float> A_transposed = transpose(A, n, n);
  std::vector<float> B_transposed = transpose(B_gpu, n, m);

  // Allocate GPU memory
  float *d_A, *d_B;
  cudaMalloc(&d_A, n * n * sizeof(float));
  cudaMalloc(&d_B, n * m * sizeof(float));

  // Copy transposed data to GPU
  cudaMemcpy(d_A, A_transposed.data(), n * n * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B_transposed.data(), n * m * sizeof(float),
             cudaMemcpyHostToDevice);

  // Perform GPU TRSM
  trsm_gpu(d_A, d_B, n, m);

  // Copy result back to host
  cudaMemcpy(B_transposed.data(), d_B, n * m * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Transpose the result back to row-major order
  B_gpu = transpose(B_transposed, m, n);

  // Compare results
  bool results_match = compare_results(B, B_gpu);

  // Print results
  std::cout << "CPU Result:" << std::endl;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      std::cout << B[i * m + j] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "\nGPU Result:" << std::endl;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      std::cout << B_gpu[i * m + j] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "\nResults " << (results_match ? "match" : "do not match")
            << std::endl;

  // Clean up
  cudaFree(d_A);
  cudaFree(d_B);

  return 0;
}