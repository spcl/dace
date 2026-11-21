// Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#if defined(__HIPCC__) || defined(WITH_HIP)
// HIP dialect: hipSOLVER mirrors cuSOLVER, so this header keeps the cuSOLVER names (see dace/libraries/hip_dialect.py)
#include <hip/hip_runtime.h>
#if __has_include(<hipsolver/hipsolver.h>)
#include <hipsolver/hipsolver.h>
#else
#include <hipsolver.h>
#endif
#define cusolverStatus_t hipsolverStatus_t
#define CUSOLVER_STATUS_SUCCESS HIPSOLVER_STATUS_SUCCESS
#define cusolverDnHandle_t hipsolverDnHandle_t
#define cusolverDnCreate hipsolverDnCreate
#define cusolverDnDestroy hipsolverDnDestroy
#else
#include <cuda_runtime.h>
#include <cusolverDn.h>
#endif

#include <cstddef>    // size_t
#include <map>
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <optional>
#include <utility>

namespace dace {

namespace lapack {

static void CheckCusolverDnError(cusolverStatus_t const& status) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error("cuSOLVER failed with error code: " +
                             std::to_string(status));
  }
}

static cusolverDnHandle_t CreateCusolverDnHandle() {
  cusolverDnHandle_t handle;
  CheckCusolverDnError(cusolverDnCreate(&handle));
  return handle;
}

/**
 * CUSOLVERDN wrapper class for DaCe. Once constructed, the class can be used to
 * get or create the CUSOLVERDN library handle (cusolverDnHandle_t). One per
 * process, like the GPU itself. The class is constructed when the CUSOLVERDN DaCe library is used.
 **/
class CusolverDnHandle {
 public:
  CusolverDnHandle() = default;
  CusolverDnHandle(CusolverDnHandle const&) = delete;

  cusolverDnHandle_t& Get() {
    if (!handle_) {
      cusolverDnHandle_t handle = CreateCusolverDnHandle();
      handle_ = handle;
    }
    return *handle_;
  }

#if defined(__HIPCC__) || defined(WITH_HIP)
  using Stream = hipStream_t;
#else
  using Stream = cudaStream_t;
#endif

  // A device buffer of at least ``bytes`` bytes for buffer ``slot`` of the calls on ``stream``, kept across calls:
  // grown on demand (the synchronizing free waits for the calls still using it), freed with the handle. Calls on
  // one stream share it, which the stream serializes; distinct streams and slots never do.
  void* Workspace(int slot, size_t bytes, Stream stream) {
    Buffer& buffer = workspaces_[{stream, slot}];
    if (bytes > buffer.bytes || buffer.data == nullptr) {
      Release(buffer);
      void* data = nullptr;
#if defined(__HIPCC__) || defined(WITH_HIP)
      const bool failed = hipMalloc(&data, bytes ? bytes : 1) != hipSuccess;
#else
      const bool failed = cudaMalloc(&data, bytes ? bytes : 1) != cudaSuccess;
#endif
      if (failed) {
        throw std::runtime_error("cuSOLVER workspace allocation of " + std::to_string(bytes) + " bytes failed");
      }
      buffer.data = data;
      buffer.bytes = bytes;
    }
    return buffer.data;
  }

  ~CusolverDnHandle() {
    for (std::map<std::pair<Stream, int>, Buffer>::iterator it = workspaces_.begin(); it != workspaces_.end(); ++it) {
      Release(it->second);
    }
    if (handle_) CheckCusolverDnError(cusolverDnDestroy(*handle_));
  }

  CusolverDnHandle& operator=(CusolverDnHandle const&) = delete;

  std::optional<cusolverDnHandle_t> handle_;

 private:
  struct Buffer {
    void* data = nullptr;
    size_t bytes = 0;
  };

  static void Release(Buffer& buffer) {
    if (buffer.data == nullptr) return;
#if defined(__HIPCC__) || defined(WITH_HIP)
    (void)hipFree(buffer.data);
#else
    (void)cudaFree(buffer.data);
#endif
    buffer = Buffer{};
  }

  std::map<std::pair<Stream, int>, Buffer> workspaces_;
};

}  // namespace lapack

}  // namespace dace
