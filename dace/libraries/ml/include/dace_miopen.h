// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#include <cstddef>    // size_t
#include <optional>
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <vector>

namespace dace {

namespace ml {

static void CheckMiopenError(miopenStatus_t const& status) {
  if (status != miopenStatusSuccess) {
    throw std::runtime_error("MIOpen failed with error code: " +
                             std::to_string(status) + " (" +
                             miopenGetErrorString(status) + ")");
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

/** Device workspace that grows on demand and is released with its owner. */
class HipWorkspace {
 public:
  HipWorkspace() = default;
  HipWorkspace(HipWorkspace const&) = delete;
  HipWorkspace& operator=(HipWorkspace const&) = delete;
  ~HipWorkspace() {
    if (ptr_) (void)hipFree(ptr_);
  }

  void* Get(size_t bytes) {
    if (bytes > size_) {
      if (ptr_) (void)hipFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
      if (bytes > 0 && hipMalloc(&ptr_, bytes) != hipSuccess) {
        throw std::runtime_error("Failed to allocate a " + std::to_string(bytes) +
                                 "-byte MIOpen workspace");
      }
      size_ = bytes;
    }
    return ptr_;
  }

 private:
  void* ptr_ = nullptr;
  size_t size_ = 0;
};

/**
 * Descriptors, algorithm choice and workspace of one convolution library node
 * (forward, backward-data or backward-weights). The descriptors are rebuilt and
 * the algorithm is searched again (miopenFind*, with the node's buffers) only
 * when the shape key changes, so that a program whose sizes are symbolic pays
 * one vector comparison per call. The owning state struct is created in
 * __dace_init, so the descriptors are created on first use.
 **/
class MiopenConvolutionPlan {
 public:
  MiopenConvolutionPlan() = default;
  MiopenConvolutionPlan(MiopenConvolutionPlan const&) = delete;
  MiopenConvolutionPlan& operator=(MiopenConvolutionPlan const&) = delete;

  ~MiopenConvolutionPlan() {
    if (!created_) return;
    miopenDestroyTensorDescriptor(input);
    miopenDestroyTensorDescriptor(output);
    miopenDestroyTensorDescriptor(bias);
    miopenDestroyTensorDescriptor(filter);
    miopenDestroyConvolutionDescriptor(conv);
  }

  /** Returns true when the descriptors and the algorithm must be (re)built. */
  bool Configure(std::vector<int> const& key) {
    if (created_ && key == key_) return false;
    if (!created_) {
      CheckMiopenError(miopenCreateTensorDescriptor(&input));
      CheckMiopenError(miopenCreateTensorDescriptor(&output));
      CheckMiopenError(miopenCreateTensorDescriptor(&bias));
      CheckMiopenError(miopenCreateTensorDescriptor(&filter));
      CheckMiopenError(miopenCreateConvolutionDescriptor(&conv));
      created_ = true;
    }
    key_ = key;
    return true;
  }

  /** Workspace of at least ``bytes`` (the search needs the size the library reports, later calls the chosen size). */
  void* Workspace(size_t bytes) { return workspace_.Get(bytes); }
  void* Workspace() { return workspace_.Get(workspace_size); }

  miopenTensorDescriptor_t input = nullptr;   // x (forward, backward-weights) or dx (backward-data)
  miopenTensorDescriptor_t output = nullptr;  // y or dy
  miopenTensorDescriptor_t bias = nullptr;
  miopenTensorDescriptor_t filter = nullptr;  // w or dw
  miopenConvolutionDescriptor_t conv = nullptr;
  int algo = 0;  // miopenConv{Fwd,BwdData,BwdWeights}Algorithm_t
  size_t workspace_size = 0;

 private:
  bool created_ = false;
  std::vector<int> key_;
  HipWorkspace workspace_;
};

}  // namespace ml

}  // namespace dace
