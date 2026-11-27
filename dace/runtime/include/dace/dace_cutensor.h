// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#if defined(__HIPCC__) || defined(WITH_HIP)
// HIP dialect: hipTensor mirrors cuTENSOR, so this header keeps the cuTENSOR names (see dace/libraries/hip_dialect.py)
#include <hip/hip_runtime.h>
#if __has_include(<hiptensor/hiptensor.h>)
#include <hiptensor/hiptensor.h>
#else
#include <hiptensor/hiptensor.hpp>
#endif
#define cutensorStatus_t hiptensorStatus_t
#define CUTENSOR_STATUS_SUCCESS HIPTENSOR_STATUS_SUCCESS
#define cutensorGetErrorString hiptensorGetErrorString
#define cutensorHandle_t hiptensorHandle_t
#define cutensorCreate hiptensorCreate
#define cutensorDestroy hiptensorDestroy
#else
#include <cuda_runtime.h>
#include <cutensor.h>
#endif

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
