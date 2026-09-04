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

#include "cub_scratch.cuh"
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

// ---------------------------------------------------------------------------------------------
// ArgMax / ArgMin, for the one libnode that emits them (``ArgReduce``'s CUDA expansion).
//
// CUB answered an arg-reduction with a single ``KeyValuePair`` output until CCCL 2.8.0 / hipCUB
// 4.0, which superseded it with two separate output iterators -- one for the extremum, one for
// its index -- and marked the pair form deprecated. Warnings are errors here and the new spelling
// does not exist on the older toolkits, so the two are selected at preprocessor time. The pair
// form's ``.key`` / ``.value`` member names are kept by the replacement buffer, so only the CUB
// call itself differs and the rest of the routine is written once.

#if (defined(HIPCUB_VERSION_MAJOR) && HIPCUB_VERSION_MAJOR >= 4) \
    || (!defined(HIPCUB_VERSION_MAJOR) && defined(CUB_VERSION) && CUB_VERSION >= 200800)
#define DACE_CUB_ARG_REDUCE_SPLIT_OUTPUTS 1
#else
#define DACE_CUB_ARG_REDUCE_SPLIT_OUTPUTS 0
#endif

// A NON-CONTIGUOUS operand does not have to be staged first. ``DeviceReduce::ArgMax`` walks an input
// ITERATOR, so an iterator that computes ``xf(base[j * stride])`` from the counting index ``j``
// presents exactly the sequence the caller means: one streaming pass, no extra allocation, no extra
// kernel. The index CUB then reports is the position in THAT sequence -- slice-local ``0 .. items-1``
// -- which is the same index the CPU expansions answer, so the node's contract does not change with
// the lowering. The contiguous, untransformed case still hands CUB a raw pointer, which is what lets
// it use vectorised loads.
//
// thrust's iterators, not cub's: cub's are deprecated from CCCL 2.8 (CUDA 12.8), warnings are errors
// here, and CCCL 3 (CUDA 13) removed them. ``reduction.h`` picks the same way for its segmented
// reduce. rocThrust ships both, so this is not a CUDA-only preference.
#if __has_include(<thrust/iterator/counting_iterator.h>)
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#define DACE_CUB_COMPAT_THRUST_ITERATORS
#endif

namespace dace {
namespace cub {

/// Per-element transforms an ``ArgReduce`` may ask for as it reads. Spelled out rather than taken
/// from ``<cmath>``: this is evaluated in device code, where the host overload set is not the one
/// that applies. ``IdentityXf`` exists so the untransformed strided case has the same iterator
/// shape as the transformed one, rather than a second code path.
struct IdentityXf {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &v) const { return v; }
};

struct AbsXf {
    /// ``v < 0 ? -v : v`` rather than a library call, so this holds for every arithmetic ``T`` the
    /// node accepts. On an unsigned ``T`` the test is never taken, which is the right answer.
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &v) const { return v < T(0) ? -v : v; }
};

/// ``j -> xf(base[j * stride])``: the whole non-contiguous read as one functor. ``stride`` is a
/// runtime value because the lift may only know it as a symbol (TSVC ``s318``'s ``inc``).
template <typename T, typename Xf>
struct StridedGather {
    const T *base;
    long long stride;
    Xf xf;
    using result_type = T;  // pre-C++11-invoke_result thrust looks this up rather than deducing it
    __host__ __device__ __forceinline__ T operator()(long long j) const { return xf(base[j * stride]); }
};

#ifdef DACE_CUB_COMPAT_THRUST_ITERATORS
template <typename T, typename Xf>
using GatherIterator = ::thrust::transform_iterator<StridedGather<T, Xf>, ::thrust::counting_iterator<long long>>;
#else
template <typename T, typename Xf>
using GatherIterator =
    ::gpucub::TransformInputIterator<T, StridedGather<T, Xf>, ::gpucub::CountingInputIterator<long long>>;
#endif

/// The iterator CUB reduces over when the operand is strided and/or transformed. ``Xf`` is named
/// explicitly by the caller; ``T`` is deduced from ``base``.
template <typename Xf, typename T>
inline GatherIterator<T, Xf> gather_iterator(const T *base, long long stride) {
#ifdef DACE_CUB_COMPAT_THRUST_ITERATORS
    return GatherIterator<T, Xf>(::thrust::counting_iterator<long long>(0), StridedGather<T, Xf>{base, stride, Xf{}});
#else
    return GatherIterator<T, Xf>(::gpucub::CountingInputIterator<long long>(0),
                                 StridedGather<T, Xf>{base, stride, Xf{}});
#endif
}

#if DACE_CUB_ARG_REDUCE_SPLIT_OUTPUTS
/// Device-side answer of an arg-reduction. ``key`` first so the block is 8-aligned whatever ``T``
/// is, and named to match the ``KeyValuePair`` it stands in for.
template <typename T>
struct ArgBuf {
    long long key;
    T value;
};
#else
template <typename T>
using ArgBuf = ::gpucub::KeyValuePair<int, T>;
#endif

/// Which extremum an :func:`arg_reduce` call looks for. ``out`` is null only on CUB's
/// size-query call, which never dereferences it.
struct ArgMaxOp {
    template <typename InIt, typename T>
    static gpuError_t call(void *scratch, size_t &needed, InIt in, ArgBuf<T> *out, long long items,
                           gpuStream_t stream) {
#if DACE_CUB_ARG_REDUCE_SPLIT_OUTPUTS
        return ::gpucub::DeviceReduce::ArgMax(scratch, needed, in, out ? &out->value : nullptr,
                                              out ? &out->key : nullptr, items, stream);
#else
        return ::gpucub::DeviceReduce::ArgMax(scratch, needed, in, out, (int)items, stream);
#endif
    }
};

struct ArgMinOp {
    template <typename InIt, typename T>
    static gpuError_t call(void *scratch, size_t &needed, InIt in, ArgBuf<T> *out, long long items,
                           gpuStream_t stream) {
#if DACE_CUB_ARG_REDUCE_SPLIT_OUTPUTS
        return ::gpucub::DeviceReduce::ArgMin(scratch, needed, in, out ? &out->value : nullptr,
                                              out ? &out->key : nullptr, items, stream);
#else
        return ::gpucub::DeviceReduce::ArgMin(scratch, needed, in, out, (int)items, stream);
#endif
    }
};

/// One arg-reduction over a device sequence, answered on the HOST.
///
/// ``in`` is any CUB-acceptable input iterator: a raw ``const T *`` for a contiguous operand, or a
/// :func:`gather_iterator` for a strided and/or transformed one. ``T`` is deduced from ``val_out``,
/// not from ``in``, so both spellings reach the same routine.
///
/// The result block sits at the front of the same ``ReduceTag`` scratch allocation as CUB's
/// workspace, pushed to the allocator's 256-byte granularity so the workspace stays aligned.
/// ``val_out`` may be null when only the index is wanted. Ties break toward the LOWER index,
/// which is the first-occurrence rule the sequential source has.
template <typename Op, typename InIt, typename T>
inline gpuError_t arg_reduce(InIt in, T *val_out, long long *idx_out, long long items, gpuStream_t stream) {
    size_t needed = 0;
    gpuError_t status = Op::call(nullptr, needed, in, (ArgBuf<T> *)nullptr, items, stream);
    if (status != gpuSuccess) return status;
    const size_t head = ((sizeof(ArgBuf<T>) + 255) / 256) * 256;
    void *scratch = get_scratch<ReduceTag>(head + needed, stream, &status);
    if (scratch == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;
    ArgBuf<T> *dev = (ArgBuf<T> *)scratch;
    status = Op::call((char *)scratch + head, needed, in, dev, items, stream);
    if (status != gpuSuccess) return status;
    ArgBuf<T> host;
    status = gpuMemcpyAsync(&host, dev, sizeof(ArgBuf<T>), gpuMemcpyDeviceToHost, stream);
    if (status != gpuSuccess) return status;
    status = gpuStreamSynchronize(stream);
    if (status != gpuSuccess) return status;
    if (val_out != nullptr) *val_out = host.value;
    *idx_out = (long long)host.key;
    return gpuSuccess;
}

}  // namespace cub
}  // namespace dace


#endif  // __DACE_CUB_COMPAT_CUH
