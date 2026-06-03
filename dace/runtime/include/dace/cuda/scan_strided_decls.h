// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// ``extern "C"`` declarations for the strided inclusive scan wrappers defined
// in :file:`dace/runtime/include/dace/cuda/scan_strided.cu`. The host ``.cpp``
// translation unit (compiled by g++) includes this header to call the
// nvcc-compiled wrappers without ever seeing a ``__global__`` symbol or the
// ``<<<>>>`` launch syntax.
//
// Suffix conventions for the dispatched dtype:
//   f64 = double   f32 = float   i64 = long long   i32 = int
// Picked at libnode-expansion time from the actual scan-buffer descriptor's
// ``dtype`` (see :class:`Scan.ExpandCUDA`).

#ifndef __DACE_CUDA_SCAN_STRIDED_DECLS_H
#define __DACE_CUDA_SCAN_STRIDED_DECLS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define _DACE_DECL_STRIDED_SCAN(SUFFIX, T)                                                                  \
    void dace_cuda_strided_inclusive_sum_##SUFFIX(const T* in, T* out, long n, long s,                      \
                                                  cudaStream_t stream);                                     \
    void dace_cuda_strided_inclusive_product_##SUFFIX(const T* in, T* out, long n, long s,                  \
                                                      cudaStream_t stream);                                 \
    void dace_cuda_strided_inclusive_min_##SUFFIX(const T* in, T* out, long n, long s,                      \
                                                  cudaStream_t stream);                                     \
    void dace_cuda_strided_inclusive_max_##SUFFIX(const T* in, T* out, long n, long s,                      \
                                                  cudaStream_t stream);

_DACE_DECL_STRIDED_SCAN(f64, double)
_DACE_DECL_STRIDED_SCAN(f32, float)
_DACE_DECL_STRIDED_SCAN(i64, long long)
_DACE_DECL_STRIDED_SCAN(i32, int)

#undef _DACE_DECL_STRIDED_SCAN

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // __DACE_CUDA_SCAN_STRIDED_DECLS_H
