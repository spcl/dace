// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Backend and toolkit portability layer for the binary-operator functors the
// ``gpucub::DeviceScan`` and ``gpucub::DeviceReduce`` libnode expansions pass into
// their host-side launchers. CCCL 13 dropped the ``gpucub::Sum`` / ``gpucub::Min``
// / ``gpucub::Max`` structs in favour of standard-library aliases. To compile
// against both toolkits, route through these macros instead of naming the
// functor directly in the libnode tasklet code.
//
// The selection is at preprocessor time so neither path costs extra at
// runtime, and there is no namespace pollution beyond the macro names.

#ifndef __DACE_CUB_COMPAT_CUH
#define __DACE_CUB_COMPAT_CUH

#include "cuda/gpucub.cuh"

#if defined(__HIPCC__) || defined(WITH_HIP)
// hipCUB keeps the functor structs CCCL 3 dropped, so the legacy spelling is the only one that
// applies here -- and the CCCL version macros below are a CUDA-only fact that says nothing about it.
#define DACE_CUB_SUM_OP ::gpucub::Sum()
#define DACE_CUB_MIN_OP ::gpucub::Min()
#define DACE_CUB_MAX_OP ::gpucub::Max()
#elif defined(CUB_MAJOR_VERSION) && CUB_MAJOR_VERSION >= 3
// CCCL 3.x (shipped with CUDA Toolkit 13+) removed the inline functor structs;
// ``cuda::std::plus`` is the supported replacement for ``gpucub::Sum``. ``min``/``max``
// remain as device-side lambdas because ``cuda::std::minimum`` / ``maximum`` are
// not part of the CCCL surface area as of 13.0.
#include <cuda/std/functional>
#define DACE_CUB_SUM_OP ::cuda::std::plus<>{}
#define DACE_CUB_MIN_OP \
    [] __device__(auto _a, auto _b) { return _a < _b ? _a : _b; }
#define DACE_CUB_MAX_OP \
    [] __device__(auto _a, auto _b) { return _a > _b ? _a : _b; }
#else
// CUB 1.x / 2.x (shipped with CUDA Toolkit 11 / 12): use the legacy structs.
#define DACE_CUB_SUM_OP ::gpucub::Sum()
#define DACE_CUB_MIN_OP ::gpucub::Min()
#define DACE_CUB_MAX_OP ::gpucub::Max()
#endif

// ``product`` was never a CUB-provided functor in any version; use a lambda.
#define DACE_CUB_MUL_OP \
    [] __device__(auto _a, auto _b) { return _a * _b; }

#endif  // __DACE_CUB_COMPAT_CUH
