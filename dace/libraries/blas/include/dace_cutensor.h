// Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include <cutensor.h>

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <unordered_map>

namespace dace {

namespace blas {

static void CheckCutensorError(cutensorStatus_t const& status) {
  if (status != CUTENSOR_STATUS_SUCCESS) {
    throw std::runtime_error("cuSPARSE failed with error code: " + std::to_string(status));
  }
}

static cutensorHandle_t* CreateCutensorHandle(int device) {
  if (cudaSetDevice(device) != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device.");
  }
  cutensorHandle_t* handle;
  CheckCutensorError(cutensorCreate(&handle));
  return handle;
}



/**
 * CUsparse wrapper class for DaCe. Once constructed, the class can be used to
 * get or create a cuSPARSE library handle (cutensorHandle_t) for a given
 * GPU ID. The class is constructed when the cuSPARSE DaCe library is used.
 **/
class CutensorHandle {
 public:
  CutensorHandle() = default;
  CutensorHandle(CutensorHandle const&) = delete;

  cutensorHandle_t* Get(int device) {
    auto f = handles_.find(device);
    if (f == handles_.end()) {
      // Lazily construct new cutensor handle if the specified key does not
      // yet exist
      cutensorHandle_t* handle = CreateCutensorHandle(device);
      f = handles_.emplace(device, handle).first;
    }
    return f->second;
  }

  ~CutensorHandle() {
    for (auto& h : handles_) {
      CheckCutensorError(cutensorDestroy(h.second));
    }
  }

  CutensorHandle& operator=(CutensorHandle const&) = delete;

  std::unordered_map<int, cutensorHandle_t*> handles_;
};

}  // namespace tensor

}  // namespace dace
