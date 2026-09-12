// Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <optional>

namespace dace {

namespace sparse {

static void CheckCusparseError(cusparseStatus_t const& status) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error("cuSPARSE failed with error code: " +
                             std::to_string(status));
  }
}

static cusparseHandle_t CreateCusparseHandle() {
  cusparseHandle_t handle;
  CheckCusparseError(cusparseCreate(&handle));
  return handle;
}

/**
 * CUsparse wrapper class for DaCe. Once constructed, the class can be used to
 * get or create the cuSPARSE library handle (cusparseHandle_t). One per
 * process, like the GPU itself. The class is constructed when the cuSPARSE DaCe library is used.
 **/
class CusparseHandle {
 public:
  CusparseHandle() = default;
  CusparseHandle(CusparseHandle const&) = delete;

  cusparseHandle_t& Get() {
    if (!handle_) {
      auto handle = CreateCusparseHandle();
      handle_ = handle;
    }
    return *handle_;
  }

  ~CusparseHandle() {
    if (handle_) CheckCusparseError(cusparseDestroy(*handle_));
  }

  CusparseHandle& operator=(CusparseHandle const&) = delete;

  std::optional<cusparseHandle_t> handle_;
};

}  // namespace sparse

}  // namespace dace
