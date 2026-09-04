// Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <optional>

namespace dace {

namespace sparse {

static void CheckHipsparseError(hipsparseStatus_t const& status) {
  if (status != HIPSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error("hipSPARSE failed with error code: " +
                             std::to_string(status));
  }
}

static hipsparseHandle_t CreateHipsparseHandle() {
  hipsparseHandle_t handle;
  CheckHipsparseError(hipsparseCreate(&handle));
  return handle;
}

/**
 * CUsparse wrapper class for DaCe. Once constructed, the class can be used to
 * get or create the hipSPARSE library handle (hipsparseHandle_t). One per
 * process, like the GPU itself. The class is constructed when the hipSPARSE DaCe library is used.
 **/
class HipsparseHandle {
 public:
  HipsparseHandle() = default;
  HipsparseHandle(HipsparseHandle const&) = delete;

  hipsparseHandle_t& Get() {
    if (!handle_) {
      auto handle = CreateHipsparseHandle();
      handle_ = handle;
    }
    return *handle_;
  }

  ~HipsparseHandle() {
    if (handle_) CheckHipsparseError(hipsparseDestroy(*handle_));
  }

  HipsparseHandle& operator=(HipsparseHandle const&) = delete;

  std::optional<hipsparseHandle_t> handle_;
};

}  // namespace sparse

}  // namespace dace
