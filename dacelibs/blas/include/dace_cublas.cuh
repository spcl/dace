#pragma once

#include "dace/cuda/cudacommon.cuh"

namespace dacelib {

namespace blas {

class CublasHelper {
  static cublasHandle_t* GlobalHandle() {
    static CublasHelper singleton;
    return singleton.handle_;
  }

 private:
  CublasHelper() { DACE_CUDA_CHECK(cublasCreate(&handle_)); }

  CublasHelper(CublasHelper const&) = delete;

  ~CublasHelper() { DACE_CUDA_CHECK(cublasDestroy(handle_)); }

  CublasHelper& operator=(CublasHelper const&) = delete;

  cublasHandle_t handle_;
};

}  // namespace blas

}  // namespace dacelib
