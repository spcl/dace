// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <optional>

namespace dace {

namespace ml {

static void CheckMiopenError(miopenStatus_t const& status) {
  if (status != miopenStatusSuccess) {
    throw std::runtime_error("MIOpen failed with error code: " +
                             std::to_string(status));
  }
}

static miopenHandle_t CreateMiopenHandle() {
  miopenHandle_t handle;
  CheckMiopenError(miopenCreate(&handle));
  return handle;
}

/**
 * MIOpen wrapper class for DaCe. Lazily creates and owns the MIOpen library
 * handle (one per process, like the GPU itself). Modeled on the cuBLAS/rocBLAS
 * handle wrappers used by the BLAS library.
 **/
class MiopenHandle {
 public:
  MiopenHandle() = default;
  MiopenHandle(MiopenHandle const&) = delete;

  miopenHandle_t& Get() {
    if (!handle_) {
      handle_ = CreateMiopenHandle();
    }
    return *handle_;
  }

  // A destructor that throws terminates the process; teardown failures have
  // nowhere left to go, so they are dropped rather than crashing.
  ~MiopenHandle() {
    if (handle_) miopenDestroy(*handle_);
  }

  MiopenHandle& operator=(MiopenHandle const&) = delete;

  std::optional<miopenHandle_t> handle_;
};

}  // namespace ml

}  // namespace dace
