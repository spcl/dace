// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
#endif  // WITH_CUDA

#endif  // __DACE_CUDAINTEROP_H
