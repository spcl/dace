// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include <cutensor.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <optional>

namespace dace {
namespace linalg {

static void CheckCuTensorError(cutensorStatus_t const& status) {
  if (status != CUTENSOR_STATUS_SUCCESS) {
    throw std::runtime_error("cuTENSOR failed with error code: " +
                             std::string(cutensorGetErrorString(status)));
  }
}

static cutensorHandle_t CreateCuTensorHandle() {
  cutensorHandle_t handle;
  CheckCuTensorError(cutensorCreate(&handle));
  return handle;
}

class CuTensorHandle {
 public:
  CuTensorHandle() = default;
  CuTensorHandle(CuTensorHandle const&) = delete;

  cutensorHandle_t& Get() {
    if (!handle_) {
      auto handle = CreateCuTensorHandle();
      handle_ = handle;
    }
    return *handle_;
  }

  ~CuTensorHandle() {
    if (handle_) cutensorDestroy(*handle_);
  }

  CuTensorHandle& operator=(CuTensorHandle const&) = delete;

  std::optional<cutensorHandle_t> handle_;
};

}  // namespace linalg
}  // namespace dace
