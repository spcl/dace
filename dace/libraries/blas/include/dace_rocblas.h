// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>  // for __float2half
#include <hip/amd_detail/amd_hip_complex.h> // for hip*Complex

#include <rocblas/rocblas.h>

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <unordered_map>

namespace dace {

namespace blas {

static void CheckRocblasError(rocblas_status const& status) {
  if (status != rocblas_status_success) {
    throw std::runtime_error("rocBLAS failed with error code: " +
                             std::to_string(status));
  }
}

static rocblas_handle CreateRocblasHandle(int device) {
  if (hipSetDevice(device) != hipSuccess) {
    throw std::runtime_error("Failed to set HIP device.");
  }
  rocblas_handle handle;
  CheckRocblasError(rocblas_create_handle(&handle));
  return handle;
}

/**
 * Class for ROCBLAS constants. Contains preallocated values for zero, one,
 * custom alpha and custom beta values.
 **/
class _RocblasConstants {
 public:
  __half const* HalfZero() const { return (__half*)zero_; }
  float const* FloatZero() const { return (float*)zero_; }
  double const* DoubleZero() const { return (double*)zero_; }
  hipComplex const* Complex64Zero() const { return (hipComplex*)zero_; }
  hipDoubleComplex const* Complex128Zero() const {
    return (hipDoubleComplex*)zero_;
  }
  __half const* HalfPone() const { return half_pone_; }
  float const* FloatPone() const { return float_pone_; }
  double const* DoublePone() const { return double_pone_; }
  hipComplex const* Complex64Pone() const { return complex64_pone_; }
  hipDoubleComplex const* Complex128Pone() const { return complex128_pone_; }

  __half* HalfAlpha() const { return (__half*)custom_alpha_; }
  float* FloatAlpha() const { return (float*)custom_alpha_; }
  double* DoubleAlpha() const { return (double*)custom_alpha_; }
  hipComplex* Complex64Alpha() const { return (hipComplex*)custom_alpha_; }
  hipDoubleComplex* Complex128Alpha() const {
    return (hipDoubleComplex*)custom_alpha_;
  }

  __half* HalfBeta() const { return (__half*)custom_beta_; }
  float* FloatBeta() const { return (float*)custom_beta_; }
  double* DoubleBeta() const { return (double*)custom_beta_; }
  hipComplex* Complex64Beta() const { return (hipComplex*)custom_beta_; }
  hipDoubleComplex* Complex128Beta() const {
    return (hipDoubleComplex*)custom_beta_;
  }

  _RocblasConstants(int device) {
    if (hipSetDevice(device) != hipSuccess) {
      throw std::runtime_error("Failed to set HIP device.");
    }
    // Allocate constant zero with the largest used size
    hipMalloc(&zero_, sizeof(hipDoubleComplex) * 1);
    hipMemset(zero_, 0, sizeof(hipDoubleComplex) * 1);

    // Allocate constant one
    hipMalloc(&half_pone_, sizeof(__half) * 1);
    __half half_pone = __float2half(1.0f);
    hipMemcpy(half_pone_, &half_pone, sizeof(__half) * 1,
               hipMemcpyHostToDevice);
    hipMalloc(&float_pone_, sizeof(float) * 1);
    float float_pone = 1.0f;
    hipMemcpy(float_pone_, &float_pone, sizeof(float) * 1,
               hipMemcpyHostToDevice);
    hipMalloc(&double_pone_, sizeof(double) * 1);
    double double_pone = 1.0;
    hipMemcpy(double_pone_, &double_pone, sizeof(double) * 1,
               hipMemcpyHostToDevice);
    hipMalloc(&complex64_pone_, sizeof(hipComplex) * 1);
    hipComplex complex64_pone = make_hipFloatComplex(1.0f, 0.0f);
    hipMemcpy(complex64_pone_, &complex64_pone, sizeof(hipComplex) * 1,
               hipMemcpyHostToDevice);
    hipMalloc(&complex128_pone_, sizeof(hipDoubleComplex) * 1);
    hipDoubleComplex complex128_pone = make_hipDoubleComplex(1.0, 0.0);
    hipMemcpy(complex128_pone_, &complex128_pone, sizeof(hipDoubleComplex) * 1,
               hipMemcpyHostToDevice);

    // Allocate custom factors and default to zero
    hipMalloc(&custom_alpha_, sizeof(hipDoubleComplex) * 1);
    hipMemset(custom_alpha_, 0, sizeof(hipDoubleComplex) * 1);
    hipMalloc(&custom_beta_, sizeof(hipDoubleComplex) * 1);
    hipMemset(custom_beta_, 0, sizeof(hipDoubleComplex) * 1);
  }

  _RocblasConstants(_RocblasConstants const&) = delete;

  ~_RocblasConstants() {
    hipFree(zero_);
    hipFree(half_pone_);
    hipFree(float_pone_);
    hipFree(double_pone_);
    hipFree(complex64_pone_);
    hipFree(complex128_pone_);
    hipFree(custom_alpha_);
    hipFree(custom_beta_);
  }

  _RocblasConstants& operator=(_RocblasConstants const&) = delete;

  void CheckError(rocblas_status const& status) {
    if (status != rocblas_status_success) {
      throw std::runtime_error("rocBLAS failed with error code: " +
                               std::to_string(status));
    }
  }

  void* zero_;
  __half* half_pone_;
  float* float_pone_;
  double* double_pone_;
  hipComplex* complex64_pone_;
  hipDoubleComplex* complex128_pone_;
  void* custom_alpha_;
  void* custom_beta_;
};

/**
 * ROCBLAS wrapper class for DaCe. Once constructed, the class can be used to
 * get or create a ROCBLAS library handle (rocblas_handle) for a given GPU ID,
 * or get pre-allocated constants (see ``_RocblasConstants`` class) for ROCBLAS
 * calls.
 * The class is constructed when the ROCBLAS DaCe library is used.
 **/
class RocblasHandle {
 public:
  RocblasHandle() = default;
  RocblasHandle(RocblasHandle const&) = delete;

  rocblas_handle& Get(int device) {
    auto f = handles_.find(device);
    if (f == handles_.end()) {
      // Lazily construct new rocBLAS handle if the specified key does not yet
      // exist
      auto handle = CreateRocblasHandle(device);
      rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
      f = handles_.emplace(device, handle).first;
    }
    return f->second;
  }

  _RocblasConstants& Constants(int device) {
    auto f = constants_.find(device);
    if (f == constants_.end()) {
      // Lazily construct new rocBLAS constants
      f = constants_.emplace(device, device).first;
    }
    return f->second;
  }

  ~RocblasHandle() {
    for (auto& h : handles_) {
      CheckRocblasError(rocblas_destroy_handle(h.second));
    }
  }

  RocblasHandle& operator=(RocblasHandle const&) = delete;

  std::unordered_map<int, rocblas_handle> handles_;
  std::unordered_map<int, _RocblasConstants> constants_;
};

}  // namespace blas

}  // namespace dace
