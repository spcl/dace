#pragma once

#include <cublas_v2.h>

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <unordered_map>

namespace dacelib {

namespace blas {

template <typename Key = size_t>
class CublasHelper {
  static cublasHandle_t& GetHandle(Key i) {
    static thread_local CublasHelper singleton;
    return singleton._GetHandle(i);
  }

 private:
  CublasHelper() = default;

  CublasHelper(CublasHelper const&) = delete;

  ~CublasHelper() {
    for (auto& h : handles_) {
      CheckError(cublasDestroy(h.second));
    }
  }

  CublasHelper& operator=(CublasHelper const&) = delete;

  cublasHandle_t& _GetHandle(Key i) {
    auto f = handles_.find(i);
    if (f == handles_.end()) {
      // Lazily construct new cuBLAS handle
      cublasHandle_t handle;
      CheckError(cublasCreate(&handle));
      f = handles_.emplace(i, handle).first;
    }
    return f->second;
  }

  void CheckError(cublasStatus_t const& status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLAS failed with error code: " +
                               std::to_string(status));
    }
  }

  std::unordered_map<Key, cublasHandle_t> handles_;
};

}  // namespace blas

}  // namespace dacelib
