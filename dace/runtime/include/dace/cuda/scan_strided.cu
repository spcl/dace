// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// CUDA-compiled host wrappers for ``dace::cuda_scan::strided_inclusive_<op>``.
// The library-node tasklet that needs a strided GPU scan lands in the SDFG's
// host ``.cpp`` translation unit (Scan libnodes sit at host schedule), and
// g++ cannot directly launch ``__global__`` kernels (``<<<>>>`` is nvcc-only,
// and ``cudaLaunchKernel`` against a ``__global__`` symbol still requires
// nvcc to materialise the function pointer). This translation unit lives in
// the ``auxiliary_sources`` list of the CUB environment so the codegen
// CMakeLists picks it up, compiles it with nvcc, and links the resulting
// object into the SDFG library; the host ``.cpp`` then calls the
// ``extern "C"`` wrappers via the declarations in
// :file:`dace/runtime/include/dace/cuda/scan_strided_decls.h`.

#include "dace/cuda/scan.cuh"

extern "C" {

#define _DACE_DEFINE_STRIDED_SCAN(SUFFIX, T)                                                            \
    void dace_cuda_strided_inclusive_sum_##SUFFIX(const T* in, T* out, long n, long s,                  \
                                                  cudaStream_t stream) {                                \
        ::dace::cuda_scan::strided_inclusive_sum<T>(in, out, n, s, stream);                             \
    }                                                                                                   \
    void dace_cuda_strided_inclusive_product_##SUFFIX(const T* in, T* out, long n, long s,              \
                                                      cudaStream_t stream) {                            \
        ::dace::cuda_scan::strided_inclusive_product<T>(in, out, n, s, stream);                         \
    }                                                                                                   \
    void dace_cuda_strided_inclusive_min_##SUFFIX(const T* in, T* out, long n, long s,                  \
                                                  cudaStream_t stream) {                                \
        ::dace::cuda_scan::strided_inclusive_min<T>(in, out, n, s, stream);                             \
    }                                                                                                   \
    void dace_cuda_strided_inclusive_max_##SUFFIX(const T* in, T* out, long n, long s,                  \
                                                  cudaStream_t stream) {                                \
        ::dace::cuda_scan::strided_inclusive_max<T>(in, out, n, s, stream);                             \
    }

_DACE_DEFINE_STRIDED_SCAN(f64, double)
_DACE_DEFINE_STRIDED_SCAN(f32, float)
_DACE_DEFINE_STRIDED_SCAN(i64, long long)
_DACE_DEFINE_STRIDED_SCAN(i32, int)

#undef _DACE_DEFINE_STRIDED_SCAN

}  // extern "C"
