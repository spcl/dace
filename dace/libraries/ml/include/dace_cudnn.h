// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstddef>    // size_t
#include <optional>
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <vector>

namespace dace {

namespace ml {

static void CheckCudnnError(cudnnStatus_t const& status) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error("cuDNN failed with error code: " +
                             std::to_string(status) + " (" +
                             cudnnGetErrorString(status) + ")");
  }
}

static cudnnHandle_t CreateCudnnHandle() {
  cudnnHandle_t handle;
  CheckCudnnError(cudnnCreate(&handle));
  return handle;
}

/**
 * cuDNN wrapper class for DaCe. Lazily creates and owns the cuDNN library
 * handle (one per process, like the GPU itself). Modeled on the cuBLAS/rocBLAS
 * handle wrappers used by the BLAS library.
 **/
class CudnnHandle {
 public:
  CudnnHandle() = default;
  CudnnHandle(CudnnHandle const&) = delete;

  cudnnHandle_t& Get() {
    if (!handle_) {
      handle_ = CreateCudnnHandle();
    }
    return *handle_;
  }

  // A destructor that throws terminates the process; teardown failures have
  // nowhere left to go, so they are dropped rather than crashing.
  ~CudnnHandle() {
    if (handle_) cudnnDestroy(*handle_);
  }

  CudnnHandle& operator=(CudnnHandle const&) = delete;

  std::optional<cudnnHandle_t> handle_;
};

/** Device workspace that grows on demand and is released with its owner. */
class CudaWorkspace {
 public:
  CudaWorkspace() = default;
  CudaWorkspace(CudaWorkspace const&) = delete;
  CudaWorkspace& operator=(CudaWorkspace const&) = delete;
  ~CudaWorkspace() {
    if (ptr_) cudaFree(ptr_);
  }

  void* Get(size_t bytes) {
    if (bytes > size_) {
      if (ptr_) cudaFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
      if (bytes > 0 && cudaMalloc(&ptr_, bytes) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate a " + std::to_string(bytes) +
                                 "-byte cuDNN workspace");
      }
      size_ = bytes;
    }
    return ptr_;
  }

 private:
  void* ptr_ = nullptr;
  size_t size_ = 0;
};

/** The algorithm cuDNN's heuristics prefer and the workspace it needs. */
struct AlgorithmChoice {
  int algo;
  size_t workspace;
};

/**
 * Runs one of the cudnnGet*Algorithm_v7 heuristics (passed as a callable taking
 * the requested count, the returned count and the result array) and returns the
 * first algorithm it reports as applicable.
 **/
template <typename Perf, typename Query>
static AlgorithmChoice SelectAlgorithm(Query query, int max_count) {
  std::vector<Perf> perf(max_count);
  int returned = 0;
  CheckCudnnError(query(max_count, &returned, perf.data()));
  for (int i = 0; i < returned; ++i) {
    if (perf[i].status == CUDNN_STATUS_SUCCESS) {
      return AlgorithmChoice{static_cast<int>(perf[i].algo), perf[i].memory};
    }
  }
  throw std::runtime_error("cuDNN found no applicable convolution algorithm");
}

/**
 * Descriptors, algorithm choice and workspace of one convolution library node
 * (forward, backward-data or backward-filter). The descriptors are rebuilt only
 * when the shape key changes, so that a program whose sizes are symbolic pays
 * one vector comparison per call. The owning state struct is created in
 * __dace_init, so the descriptors are created on first use.
 **/
class CudnnConvolutionPlan {
 public:
  CudnnConvolutionPlan() = default;
  CudnnConvolutionPlan(CudnnConvolutionPlan const&) = delete;
  CudnnConvolutionPlan& operator=(CudnnConvolutionPlan const&) = delete;

  ~CudnnConvolutionPlan() {
    if (!created_) return;
    cudnnDestroyTensorDescriptor(input);
    cudnnDestroyTensorDescriptor(output);
    cudnnDestroyTensorDescriptor(bias);
    cudnnDestroyFilterDescriptor(filter);
    cudnnDestroyConvolutionDescriptor(conv);
    cudnnDestroyActivationDescriptor(activation);
  }

  /** Returns true when the descriptors and the algorithm must be (re)built. */
  bool Configure(std::vector<int> const& key) {
    if (created_ && key == key_) return false;
    if (!created_) {
      CheckCudnnError(cudnnCreateTensorDescriptor(&input));
      CheckCudnnError(cudnnCreateTensorDescriptor(&output));
      CheckCudnnError(cudnnCreateTensorDescriptor(&bias));
      CheckCudnnError(cudnnCreateFilterDescriptor(&filter));
      CheckCudnnError(cudnnCreateConvolutionDescriptor(&conv));
      CheckCudnnError(cudnnCreateActivationDescriptor(&activation));
      created_ = true;
    }
    key_ = key;
    fused = true;
    return true;
  }

  void* Workspace() { return workspace_.Get(workspace_size); }

  /** A zero bias of ``bytes`` bytes (the fused convolution+bias+activation call requires a bias). */
  const void* ZeroBias(size_t bytes) {
    if (bytes > zero_bias_bytes_) {
      void* ptr = zero_bias_.Get(bytes);
      if (cudaMemset(ptr, 0, bytes) != cudaSuccess) {
        throw std::runtime_error("Failed to initialize the zero bias of a fused convolution");
      }
      zero_bias_bytes_ = bytes;
    }
    return zero_bias_.Get(bytes);
  }

  cudnnTensorDescriptor_t input = nullptr;   // x (forward, backward-filter) or dx (backward-data)
  cudnnTensorDescriptor_t output = nullptr;  // y or dy
  cudnnTensorDescriptor_t bias = nullptr;
  cudnnFilterDescriptor_t filter = nullptr;  // w or dw
  cudnnConvolutionDescriptor_t conv = nullptr;
  cudnnActivationDescriptor_t activation = nullptr;
  int algo = 0;  // cudnnConvolution{Fwd,BwdData,BwdFilter}Algo_t
  size_t workspace_size = 0;
  // Whether the fused convolution+bias+activation call applies; cleared when cuDNN reports it unsupported for
  // this configuration, after which the separate calls are used.
  bool fused = true;

 private:
  bool created_ = false;
  std::vector<int> key_;
  CudaWorkspace workspace_;
  CudaWorkspace zero_bias_;
  size_t zero_bias_bytes_ = 0;
};

}  // namespace ml

}  // namespace dace
