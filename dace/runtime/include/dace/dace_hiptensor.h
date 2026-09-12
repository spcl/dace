// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

// hipTensor mirrors the cuTENSOR v2 surface call for call -- descriptor, permutation, plan
// preference, plan, execute, destroy -- so the expansion body is shared and only the names differ.
// The one genuine divergence is the data type: cuTENSOR takes the CUDA-wide ``cudaDataType``,
// hipTensor takes its own ``hiptensorDataType_t`` (``HIPTENSOR_R_32F``, not ``HIP_R_32F``), which
// is why the type map lives per environment rather than being shared.
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <optional>

namespace dace {
namespace linalg {

static void CheckHipTensorError(hiptensorStatus_t const& status) {
  if (status != HIPTENSOR_STATUS_SUCCESS) {
    throw std::runtime_error("hipTensor failed with error code: " +
                             std::string(hiptensorGetErrorString(status)));
  }
}

static hiptensorHandle_t CreateHipTensorHandle() {
  hiptensorHandle_t handle;
  CheckHipTensorError(hiptensorCreate(&handle));
  return handle;
}

class HipTensorHandle {
 public:
  HipTensorHandle() = default;
  HipTensorHandle(HipTensorHandle const&) = delete;

  hiptensorHandle_t& Get() {
    if (!handle_) {
      auto handle = CreateHipTensorHandle();
      handle_ = handle;
    }
    return *handle_;
  }

  ~HipTensorHandle() {
    if (handle_) hiptensorDestroy(*handle_);
  }

  HipTensorHandle& operator=(HipTensorHandle const&) = delete;

  std::optional<hiptensorHandle_t> handle_;
};

}  // namespace linalg
}  // namespace dace
