// Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <unordered_map>

namespace dace {

namespace sparse {

static void CheckCusparseError(cusparseStatus_t const& status) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error("cuSPARSE failed with error code: " + std::to_string(status));
  }
}

static cusparseHandle_t CreateCusparseHandle(int device) {
  if (cudaSetDevice(device) != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device.");
  }
  cusparseHandle_t handle;
  CheckCusparseError(cusparseCreate(&handle));
  return handle;
}

/**
 * CUsparse wrapper class for DaCe. Once constructed, the class can be used to
 * get or create a cuSPARSE library handle (cusparseHandle_t) for a given
 * GPU ID. The class is constructed when the cuSPARSE DaCe library is used.
 **/
class CusparseHandle {
 public:
  CusparseHandle() = default;
  CusparseHandle(CusparseHandle const&) = delete;

  cusparseHandle_t& Get(int device) {
    auto f = handles_.find(device);
    if (f == handles_.end()) {
      // Lazily construct new cusparse handle if the specified key does not
      // yet exist
      auto handle = CreateCusparseHandle(device);
      f = handles_.emplace(device, handle).first;
    }
    return f->second;
  }

  ~CusparseHandle() {
    for (auto& h : handles_) {
      CheckCusparseError(cusparseDestroy(h.second));
    }
  }

  CusparseHandle& operator=(CusparseHandle const&) = delete;

  std::unordered_map<int, cusparseHandle_t> handles_;
};

}  // namespace sparse

}  // namespace dace
