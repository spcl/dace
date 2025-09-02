// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <unordered_map>

namespace dace {

namespace blas {

static void CheckCublasError(cublasStatus_t const& status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS failed with error code: " +
                             std::to_string(status));
  }
}

static cublasHandle_t CreateCublasHandle(int device) {
  if (device >= 0) {
    if (cudaSetDevice(device) != cudaSuccess) {
      throw std::runtime_error("Failed to set CUDA device.");
    }
  }
  cublasHandle_t handle;
  CheckCublasError(cublasCreate(&handle));
  return handle;
}

/**
 * Class for CUBLAS constants. Contains preallocated values for zero, one,
 * custom alpha and custom beta values.
 **/
class _CublasConstants {
 public:
  __half const* HalfZero() const { return (__half*)zero_; }
  float const* FloatZero() const { return (float*)zero_; }
  double const* DoubleZero() const { return (double*)zero_; }
  cuComplex const* Complex64Zero() const { return (cuComplex*)zero_; }
  cuDoubleComplex const* Complex128Zero() const {
    return (cuDoubleComplex*)zero_;
  }
  __half const* HalfPone() const { return half_pone_; }
  float const* FloatPone() const { return float_pone_; }
  double const* DoublePone() const { return double_pone_; }
  cuComplex const* Complex64Pone() const { return complex64_pone_; }
  cuDoubleComplex const* Complex128Pone() const { return complex128_pone_; }

  __half* HalfAlpha() const { return (__half*)custom_alpha_; }
  float* FloatAlpha() const { return (float*)custom_alpha_; }
  double* DoubleAlpha() const { return (double*)custom_alpha_; }
  cuComplex* Complex64Alpha() const { return (cuComplex*)custom_alpha_; }
  cuDoubleComplex* Complex128Alpha() const {
    return (cuDoubleComplex*)custom_alpha_;
  }

  __half* HalfBeta() const { return (__half*)custom_beta_; }
  float* FloatBeta() const { return (float*)custom_beta_; }
  double* DoubleBeta() const { return (double*)custom_beta_; }
  cuComplex* Complex64Beta() const { return (cuComplex*)custom_beta_; }
  cuDoubleComplex* Complex128Beta() const {
    return (cuDoubleComplex*)custom_beta_;
  }

  _CublasConstants(int device) {
    if (device >= 0) {
      if (cudaSetDevice(device) != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device.");
      }
    }
    // Allocate constant zero with the largest used size
    cudaMalloc(&zero_, sizeof(cuDoubleComplex) * 1);
    cudaMemset(zero_, 0, sizeof(cuDoubleComplex) * 1);

    // Allocate constant one
    cudaMalloc(&half_pone_, sizeof(__half) * 1);
    __half half_pone = __float2half(1.0f);
    cudaMemcpy(half_pone_, &half_pone, sizeof(__half) * 1,
               cudaMemcpyHostToDevice);
    cudaMalloc(&float_pone_, sizeof(float) * 1);
    float float_pone = 1.0f;
    cudaMemcpy(float_pone_, &float_pone, sizeof(float) * 1,
               cudaMemcpyHostToDevice);
    cudaMalloc(&double_pone_, sizeof(double) * 1);
    double double_pone = 1.0;
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

    // Allocate custom factors and default to zero
    cudaMalloc(&custom_alpha_, sizeof(cuDoubleComplex) * 1);
    cudaMemset(custom_alpha_, 0, sizeof(cuDoubleComplex) * 1);
    cudaMalloc(&custom_beta_, sizeof(cuDoubleComplex) * 1);
    cudaMemset(custom_beta_, 0, sizeof(cuDoubleComplex) * 1);
  }

  _CublasConstants(_CublasConstants const&) = delete;

  ~_CublasConstants() {
    cudaFree(zero_);
    cudaFree(half_pone_);
    cudaFree(float_pone_);
    cudaFree(double_pone_);
    cudaFree(complex64_pone_);
    cudaFree(complex128_pone_);
    cudaFree(custom_alpha_);
    cudaFree(custom_beta_);
  }

  _CublasConstants& operator=(_CublasConstants const&) = delete;

  void CheckError(cublasStatus_t const& status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLAS failed with error code: " +
                               std::to_string(status));
    }
  }

  void* zero_;
  __half* half_pone_;
  float* float_pone_;
  double* double_pone_;
  cuComplex* complex64_pone_;
  cuDoubleComplex* complex128_pone_;
  void* custom_alpha_;
  void* custom_beta_;
};

/**
 * CUBLAS wrapper class for DaCe. Once constructed, the class can be used to
 * get or create a CUBLAS library handle (cublasHandle_t) for a given GPU ID,
 * or get pre-allocated constants (see ``_CublasConstants`` class) for CUBLAS
 * calls.
 * The class is constructed when the CUBLAS DaCe library is used.
 **/
class CublasHandle {
 public:
  CublasHandle() = default;
  CublasHandle(CublasHandle const&) = delete;

  cublasHandle_t& Get(int device) {
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

  _CublasConstants& Constants(int device) {
    auto f = constants_.find(device);
    if (f == constants_.end()) {
      // Lazily construct new cuBLAS constants
      f = constants_.emplace(device, device).first;
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
  std::unordered_map<int, _CublasConstants> constants_;
};

}  // namespace blas

}  // namespace dace
