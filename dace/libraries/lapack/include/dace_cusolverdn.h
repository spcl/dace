// Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <unordered_map>

namespace dace {

namespace lapack {

static void CheckCusolverDnError(cusolverStatus_t const& status) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error("cuSOLVER failed with error code: " +
                             std::to_string(status));
  }
}

static cusolverDnHandle_t CreateCusolverDnHandle(int device) {
  if (cudaSetDevice(device) != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device.");
  }
  cusolverDnHandle_t handle;
  CheckCusolverDnError(cusolverDnCreate(&handle));
  return handle;
}

/**
 * CUSOLVERDN wrapper class for DaCe. Once constructed, the class can be used to
 * get or create a CUSOLVERDN library handle (cusolverDnHandle_t) for a given
 * GPU ID. The class is constructed when the CUSOLVERDN DaCe library is used.
 **/
class CusolverDnHandle {
 public:
  CusolverDnHandle() = default;
  CusolverDnHandle(CusolverDnHandle const&) = delete;

  cusolverDnHandle_t& Get(int device) {
    auto f = handles_.find(device);
    if (f == handles_.end()) {
      // Lazily construct new cuSolverDn handle if the specified key does not
      // yet exist
      auto handle = CreateCusolverDnHandle(device);
      f = handles_.emplace(device, handle).first;
    }
    return f->second;
  }

  ~CusolverDnHandle() {
    for (auto& h : handles_) {
      CheckCusolverDnError(cusolverDnDestroy(h.second));
    }
  }

  CusolverDnHandle& operator=(CusolverDnHandle const&) = delete;

  std::unordered_map<int, cusolverDnHandle_t> handles_;
};

}  // namespace lapack

}  // namespace dace
