// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>

#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <optional>

namespace dace {

namespace ml {

static void CheckCudnnError(cudnnStatus_t const& status) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error("cuDNN failed with error code: " +
                             std::to_string(status));
  }
}

static cudnnHandle_t CreateCudnnHandle() {
  cudnnHandle_t handle;
  CheckCudnnError(cudnnCreate(&handle));
  return handle;
}

/**
 * cuDNN wrapper class for DaCe. Lazily creates and owns the cuDNN library
 * handle (one per process, like the GPU itself). Modeled on the cuBLAS/rocBLAS
 * handle wrappers used by the BLAS library.
 **/
class CudnnHandle {
 public:
  CudnnHandle() = default;
  CudnnHandle(CudnnHandle const&) = delete;

  cudnnHandle_t& Get() {
    if (!handle_) {
      handle_ = CreateCudnnHandle();
    }
    return *handle_;
  }

  // A destructor that throws terminates the process; teardown failures have
  // nowhere left to go, so they are dropped rather than crashing.
  ~CudnnHandle() {
    if (handle_) cudnnDestroy(*handle_);
  }

  CudnnHandle& operator=(CudnnHandle const&) = delete;

  std::optional<cudnnHandle_t> handle_;
};

}  // namespace ml

}  // namespace dace
