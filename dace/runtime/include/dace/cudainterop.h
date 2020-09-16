// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_CUDAINTEROP_H
#define __DACE_CUDAINTEROP_H

#ifdef WITH_CUDA
  #if defined(__HIPCC__) || defined(WITH_HIP)
    #include <hip/hip_runtime.h>
    #define gpuLaunchKernel hipLaunchKernel
    #define gpuMalloc hipMalloc
    #define gpuMemset hipMemset
    #define gpuFree hipFree
  #else
    #include <cuda_runtime.h>
    #define gpuLaunchKernel cudaLaunchKernel
    #define gpuMalloc cudaMalloc
    #define gpuMemset cudaMemset
    #define gpuFree cudaFree
  #endif

  #include "cuda/cudacommon.cuh"
#else

// CUDA interoperability (defining external functions without having to include
// cuda_runtime.h)
typedef int cudaError_t;
typedef void *cudaStream_t;
typedef void *cudaEvent_t;
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

#include "cuda/cudacommon.cuh"

extern "C"
{
    cudaError_t cudaMalloc(void **devPtr, size_t size);
    cudaError_t cudaFree(void *devPtr);
    cudaError_t cudaMallocHost(void **devPtr, size_t size);
    cudaError_t cudaFreeHost(void *devPtr);
    cudaError_t cudaMemcpy(void *dst, const void *src,
                           size_t count,
                           enum cudaMemcpyKind kind);
    cudaError_t cudaMemcpyAsync(void *dst, const void *src,
                                size_t count,
                                enum cudaMemcpyKind kind,
                                cudaStream_t stream = 0);
    cudaError_t cudaMemcpy2D(void *  dst,
                             size_t  dpitch,
                             const void *  src,
                             size_t  spitch,
                             size_t  width,
                             size_t  height,
                             enum cudaMemcpyKind  kind);
    cudaError_t cudaMemcpy2DAsync(void *  dst,
                             size_t  dpitch,
                             const void *  src,
                             size_t  spitch,
                             size_t  width,
                             size_t  height,
                             enum cudaMemcpyKind  kind,
                             cudaStream_t stream = 0);
    cudaError_t cudaMemsetAsync(void *dst, int value,
                                size_t count,
                                cudaStream_t stream = 0);
    cudaError_t cudaStreamSynchronize(cudaStream_t stream);
    cudaError_t cudaGetLastError(void);
    cudaError_t cudaDeviceSynchronize(void);
}


template<typename T> cudaError_t cudaMalloc(T **devPtr, size_t size) {
    return cudaMalloc((void **)devPtr, size);
}
template<typename T> cudaError_t cudaFree(T *devPtr) {
    return cudaFree((void *)devPtr);
}
template<typename T> cudaError_t cudaMallocHost(T **devPtr, size_t size) {
    return cudaMallocHost((void **)devPtr, size);
}
template<typename T> cudaError_t cudaFreeHost(T *devPtr) {
    return cudaFreeHost((void *)devPtr);
}

template<typename T> cudaError_t cudaMemcpy(T *dst, const void *src,
                                            size_t count,
                                            enum cudaMemcpyKind kind) {
    return cudaMemcpy((void *)dst, src, count, kind);
}
template<typename T> cudaError_t cudaMemcpyAsync(T *dst, const void *src,
                                                 size_t count,
                                                 enum cudaMemcpyKind kind,
                                                 cudaStream_t stream = 0) {
    return cudaMemcpyAsync((void *)dst, src, count, kind, stream);
}

template<typename T> cudaError_t cudaMemcpy2D(T *  dst,
                                              size_t  dpitch,
                                              const void *  src,
                                              size_t  spitch,
                                              size_t  width,
                                              size_t  height,
                                              enum cudaMemcpyKind  kind) {
    return cudaMemcpy2D((void*)dst, dpitch, src, spitch, width, height, kind);
}
template<typename T> cudaError_t cudaMemcpy2DAsync(T *  dst,
                                                   size_t  dpitch,
                                                   const void *  src,
                                                   size_t  spitch,
                                                   size_t  width,
                                                   size_t  height,
                                                   enum cudaMemcpyKind  kind,
                                                   cudaStream_t stream = 0) {
    return cudaMemcpy2DAsync((void*)dst, dpitch, src, spitch, width, height, kind, stream);
}
template<typename T> cudaError_t cudaMemsetAsync(T *dst, int value,
                                                 size_t count,
                                                 cudaStream_t stream = 0) {
    return cudaMemsetAsync((void *)dst, value, count, stream);
}

#endif  // WITH_CUDA

#endif  // __DACE_CUDAINTEROP_H
