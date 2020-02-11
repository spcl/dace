#pragma once

#include <cublas_v2.h>

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <unordered_map>

namespace dace {

namespace blas {

void CheckCublasError(cublasStatus_t const& status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS failed with error code: " +
                             std::to_string(status));
  }
}

cublasHandle_t CreateCublasHandle(int device) {
  if (cudaSetDevice(device) != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device.");
  }
  cublasHandle_t handle;
  CheckCublasError(cublasCreate(&handle));
  return handle;
}

class CublasHandle {
 public:
  /// Returns the singleton instance associated with the current thread.
  static cublasHandle_t& Get(int device) {
    static thread_local CublasHandle singleton;
    return singleton._Get(device);
  }

 private:
  CublasHandle() = default;
  CublasHandle(CublasHandle const&) = delete;

  cublasHandle_t& _Get(int device) {
    auto f = handles_.find(device);
    if (f == handles_.end()) {
      // Lazily construct new cuBLAS handle if the specified key does not yet
      // exist
      auto handle = CreateCublasHandle(device);
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
      f = handles_.emplace(device, handle).first;
    }
    return f->second;
  }

  ~CublasHandle() {
    for (auto& h : handles_) {
      CheckCublasError(cublasDestroy(h.second));
    }
  }

  CublasHandle& operator=(CublasHandle const&) = delete;

  std::unordered_map<int, cublasHandle_t> handles_;
};

namespace {

class _CublasConstants {
 public:
  float const* FloatZero() const { return float_zero_; }
  double const* DoubleZero() const { return double_zero_; }
  cuComplex const* Complex64Zero() const { return complex64_zero_; }
  cuDoubleComplex const* Complex128Zero() const { return complex128_zero_; }
  float const* FloatPone() const { return float_pone_; }
  double const* DoublePone() const { return double_pone_; }
  cuComplex const* Complex64Pone() const { return complex64_pone_; }
  cuDoubleComplex const* Complex128Pone() const { return complex128_pone_; }

  _CublasConstants(int device) {
    if (cudaSetDevice(device) != cudaSuccess) {
      throw std::runtime_error("Failed to set CUDA device.");
    }
    // Allocate constant zero
    cudaMalloc(&float_zero_, sizeof(float) * 1);
    float float_zero = 0.0f;
    cudaMemcpy(float_zero_, &float_zero, sizeof(float) * 1,
               cudaMemcpyHostToDevice);
    cudaMalloc(&double_zero_, sizeof(double) * 1);
    double double_zero = 0.0;
    cudaMemcpy(double_zero_, &double_zero, sizeof(double) * 1,
               cudaMemcpyHostToDevice);
    cudaMalloc(&complex64_zero_, sizeof(cuComplex) * 1);
    cuComplex complex64_zero = make_cuComplex(0.0f, 0.0f);
    cudaMemcpy(complex64_zero_, &complex64_zero, sizeof(cuComplex) * 1,
               cudaMemcpyHostToDevice);
    cudaMalloc(&complex128_zero_, sizeof(cuDoubleComplex) * 1);
    cuDoubleComplex complex128_zero = make_cuDoubleComplex(0.0, 0.0);
    cudaMemcpy(complex128_zero_, &complex128_zero, sizeof(cuDoubleComplex) * 1,
               cudaMemcpyHostToDevice);
    // Allocate constant one
    cudaMalloc(&float_pone_, sizeof(float) * 1);
    float float_pone = 1.0f;
    cudaMemcpy(float_pone_, &float_pone, sizeof(float) * 1,
               cudaMemcpyHostToDevice);
    cudaMalloc(&double_pone_, sizeof(double) * 1);
    double double_pone = 0.0;
    cudaMemcpy(double_pone_, &double_pone, sizeof(double) * 1,
               cudaMemcpyHostToDevice);
    cudaMalloc(&complex64_pone_, sizeof(cuComplex) * 1);
    cuComplex complex64_pone = make_cuComplex(1.0f, 0.0f);
    cudaMemcpy(complex64_pone_, &complex64_pone, sizeof(cuComplex) * 1,
               cudaMemcpyHostToDevice);
    cudaMalloc(&complex128_pone_, sizeof(cuDoubleComplex) * 1);
    cuDoubleComplex complex128_pone = make_cuDoubleComplex(1.0, 0.0);
    cudaMemcpy(complex128_pone_, &complex128_pone, sizeof(cuDoubleComplex) * 1,
               cudaMemcpyHostToDevice);
  }

  _CublasConstants(_CublasConstants const&) = delete;

  ~_CublasConstants() {
    cudaFree(float_zero_);
    cudaFree(double_zero_);
    cudaFree(complex64_zero_);
    cudaFree(complex128_zero_);
    cudaFree(float_pone_);
    cudaFree(double_pone_);
    cudaFree(complex64_pone_);
    cudaFree(complex128_pone_);
  }

  _CublasConstants& operator=(_CublasConstants const&) = delete;

  void CheckError(cublasStatus_t const& status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLAS failed with error code: " +
                               std::to_string(status));
    }
  }

  float* float_zero_;
  double* double_zero_;
  cuComplex* complex64_zero_;
  cuDoubleComplex* complex128_zero_;
  float* float_pone_;
  double* double_pone_;
  cuComplex* complex64_pone_;
  cuDoubleComplex* complex128_pone_;
};

}  // namespace

class CublasConstants {
 public:
  /// Returns the constants instance associated with the specified device
  static _CublasConstants& Get(int device) {
    static CublasConstants singleton;
    return singleton._Get(device);
  }

 private:
  _CublasConstants& _Get(int device) {
    auto f = constants_.find(device);
    if (f == constants_.end()) {
      // Lazily construct new cuBLAS constants
      f = constants_.emplace(device, device).first;
    }
    return f->second;
  }

  std::unordered_map<int, _CublasConstants> constants_;
};

}  // namespace blas

}  // namespace dace 
