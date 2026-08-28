// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <hip/hip_complex.h>  // for hip*Complex; the public header, not the amd_detail one
#include <hip/hip_fp16.h>                    // for __float2half
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>  // ROCm 4.5+ layout; the flat <rocblas.h> is gone by ROCm 7

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <optional>

namespace dace {

namespace blas {

static void CheckRocblasError(rocblas_status const& status) {
  if (status != rocblas_status_success) {
    throw std::runtime_error("rocBLAS failed with error code: " +
                             std::to_string(status));
  }
}

static rocblas_handle CreateRocblasHandle() {
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

  _RocblasConstants() {
    // Allocate constant zero with the largest used size
    (void)hipMalloc(&zero_, sizeof(hipDoubleComplex) * 1);
    (void)hipMemset(zero_, 0, sizeof(hipDoubleComplex) * 1);

    // Allocate constant one
    (void)hipMalloc(&half_pone_, sizeof(__half) * 1);
    __half half_pone = __float2half(1.0f);
    (void)hipMemcpy(half_pone_, &half_pone, sizeof(__half) * 1,
                    hipMemcpyHostToDevice);
    (void)hipMalloc(&float_pone_, sizeof(float) * 1);
    float float_pone = 1.0f;
    (void)hipMemcpy(float_pone_, &float_pone, sizeof(float) * 1,
                    hipMemcpyHostToDevice);
    (void)hipMalloc(&double_pone_, sizeof(double) * 1);
    double double_pone = 1.0;
    (void)hipMemcpy(double_pone_, &double_pone, sizeof(double) * 1,
                    hipMemcpyHostToDevice);
    (void)hipMalloc(&complex64_pone_, sizeof(hipComplex) * 1);
    hipComplex complex64_pone = make_hipFloatComplex(1.0f, 0.0f);
    (void)hipMemcpy(complex64_pone_, &complex64_pone, sizeof(hipComplex) * 1,
                    hipMemcpyHostToDevice);
    (void)hipMalloc(&complex128_pone_, sizeof(hipDoubleComplex) * 1);
    hipDoubleComplex complex128_pone = make_hipDoubleComplex(1.0, 0.0);
    (void)hipMemcpy(complex128_pone_, &complex128_pone,
                    sizeof(hipDoubleComplex) * 1, hipMemcpyHostToDevice);

    // Allocate custom factors and default to zero
    (void)hipMalloc(&custom_alpha_, sizeof(hipDoubleComplex) * 1);
    (void)hipMemset(custom_alpha_, 0, sizeof(hipDoubleComplex) * 1);
    (void)hipMalloc(&custom_beta_, sizeof(hipDoubleComplex) * 1);
    (void)hipMemset(custom_beta_, 0, sizeof(hipDoubleComplex) * 1);
  }

  _RocblasConstants(_RocblasConstants const&) = delete;

  ~_RocblasConstants() {
    (void)hipFree(zero_);
    (void)hipFree(half_pone_);
    (void)hipFree(float_pone_);
    (void)hipFree(double_pone_);
    (void)hipFree(complex64_pone_);
    (void)hipFree(complex128_pone_);
    (void)hipFree(custom_alpha_);
    (void)hipFree(custom_beta_);
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
 * get or create the ROCBLAS library handle (rocblas_handle), or get pre-allocated
 * constants (see ``_RocblasConstants`` class) for ROCBLAS calls. One per process,
 * like the GPU itself.
 * The class is constructed when the ROCBLAS DaCe library is used.
 **/
class RocblasHandle {
 public:
  RocblasHandle() = default;
  RocblasHandle(RocblasHandle const&) = delete;

  rocblas_handle& Get() {
    if (!handle_) {
      auto handle = CreateRocblasHandle();
      rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
      handle_ = handle;
    }
    return *handle_;
  }

  _RocblasConstants& Constants() {
    if (!constants_) constants_.emplace();
    return *constants_;
  }

  // A destructor that throws terminates the process. Teardown failures have nowhere left to go, so
  // they are dropped rather than turned into a crash that hides whatever the program computed.
  ~RocblasHandle() {
    if (handle_) CheckRocblasError(rocblas_destroy_handle(*handle_));
  }

  RocblasHandle& operator=(RocblasHandle const&) = delete;

  std::optional<rocblas_handle> handle_;
  std::optional<_RocblasConstants> constants_;
};

}  // namespace blas

}  // namespace dace
