// Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <optional>

namespace dace {

namespace lapack {

static void CheckCusolverDnError(cusolverStatus_t const& status) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error("cuSOLVER failed with error code: " +
                             std::to_string(status));
  }
}

static cusolverDnHandle_t CreateCusolverDnHandle() {
  cusolverDnHandle_t handle;
  CheckCusolverDnError(cusolverDnCreate(&handle));
  return handle;
}

/**
 * CUSOLVERDN wrapper class for DaCe. Once constructed, the class can be used to
 * get or create the CUSOLVERDN library handle (cusolverDnHandle_t). One per
 * process, like the GPU itself. The class is constructed when the CUSOLVERDN DaCe library is used.
 **/
class CusolverDnHandle {
 public:
  CusolverDnHandle() = default;
  CusolverDnHandle(CusolverDnHandle const&) = delete;

  cusolverDnHandle_t& Get() {
    if (!handle_) {
      auto handle = CreateCusolverDnHandle();
      handle_ = handle;
    }
    return *handle_;
  }

  ~CusolverDnHandle() {
    if (handle_) CheckCusolverDnError(cusolverDnDestroy(*handle_));
  }

  CusolverDnHandle& operator=(CusolverDnHandle const&) = delete;

  std::optional<cusolverDnHandle_t> handle_;
};

}  // namespace lapack

}  // namespace dace
