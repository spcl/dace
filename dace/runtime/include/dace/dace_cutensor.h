// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include <cutensor.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace dace {
namespace linalg {

static void CheckCuTensorError(cutensorStatus_t const& status) {
    if (status != CUTENSOR_STATUS_SUCCESS) {
        throw std::runtime_error("cuTENSOR failed with error code: " +
                                 std::string(cutensorGetErrorString(status)));
    }
}

static cutensorHandle_t CreateCuTensorHandle(int device) {
    if (device >= 0) {
        if (cudaSetDevice(device) != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA device.");
        }
    }
    cutensorHandle_t handle;
    CheckCuTensorError(cutensorCreate(&handle));
    return handle;
}

class CuTensorHandle {
  public:
    CuTensorHandle() = default;
    CuTensorHandle(CuTensorHandle const&) = delete;

    cutensorHandle_t& Get(int device) {
        auto f = handles_.find(device);
        if (f == handles_.end()) {
            // Lazily construct new cuTENSOR handle if the specified key does
            // not yet exist
            auto handle = CreateCuTensorHandle(device);
            f = handles_.emplace(device, handle).first;
        }
        return f->second;
    }

    ~CuTensorHandle() {
        for (auto& h : handles_) {
            cutensorDestroy(h.second);
        }
    }

    CuTensorHandle& operator=(CuTensorHandle const&) = delete;

    std::unordered_map<int, cutensorHandle_t> handles_;
};

}  // namespace linalg
}  // namespace dace
