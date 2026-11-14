// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Portability macros for the CUB/hipCUB binary-operator functors (CCCL 13 dropped ``gpucub::Sum``/``Min``/
// ``Max``). Use these macros instead of naming a functor in libnode tasklet code.

#ifndef __DACE_CUB_COMPAT_CUH
#define __DACE_CUB_COMPAT_CUH

#include <iterator>

#include "arg_reduction.h"
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
// use the ``::dace::cub`` functors below because ``cuda::std::minimum`` / ``maximum`` are
// not part of the CCCL surface area as of 13.0.
#include <cuda/std/functional>
#define DACE_CUB_SUM_OP \
  ::cuda::std::plus<> {}
#define DACE_CUB_MIN_OP \
  ::dace::cub::Minimum {}
#define DACE_CUB_MAX_OP \
  ::dace::cub::Maximum {}
#else
// CUB 1.x / 2.x (shipped with CUDA Toolkit 11 / 12): use the legacy structs.
#define DACE_CUB_SUM_OP ::gpucub::Sum()
#define DACE_CUB_MIN_OP ::gpucub::Min()
#define DACE_CUB_MAX_OP ::gpucub::Max()
#endif

#define DACE_CUB_MUL_OP \
  ::dace::cub::Multiplies {}

// ArgMax / ArgMin, for the one libnode that emits them (``ArgReduce``'s CUDA expansion).
//
// CUB answered an arg-reduction with a single ``KeyValuePair`` output until CCCL 2.8.0 / hipCUB
// 4.0, which superseded it with two separate output iterators -- one for the extremum, one for
// its index -- and marked the pair form deprecated. Warnings are errors here and the new spelling
// does not exist on the older toolkits, so the two are selected at preprocessor time. The pair
// form's ``.key`` / ``.value`` member names are kept by the replacement buffer, so only the CUB
// call itself differs and the rest of the routine is written once.

#if (defined(HIPCUB_VERSION_MAJOR) && HIPCUB_VERSION_MAJOR >= 4) || \
    (!defined(HIPCUB_VERSION_MAJOR) && defined(CUB_VERSION) && CUB_VERSION >= 200800)
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
namespace dace {
namespace cub {

/// Per-element transforms an ``ArgReduce`` may ask for as it reads. Spelled out rather than taken
/// from ``<cmath>``: this is evaluated in device code, where the host overload set is not the one
/// that applies. ``IdentityXf`` exists so the untransformed strided case has the same iterator
/// shape as the transformed one, rather than a second code path.
struct IdentityXf {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& v) const {
    return v;
  }
};

struct AbsXf {
  /// ``v < 0 ? -v : v`` rather than a library call, so this holds for every arithmetic ``T`` the
  /// node accepts. On an unsigned ``T`` the test is never taken, which is the right answer.
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& v) const {
    return v < T(0) ? -v : v;
  }
};

/// Functors, not lambdas: nvcc rejects a ``__device__`` lambda without ``--extended-lambda``.
struct Multiplies {
  template <typename A, typename B>
  __host__ __device__ __forceinline__ auto operator()(const A& a, const B& b) const {
    return a * b;
  }
};

struct Minimum {
  template <typename A, typename B>
  __host__ __device__ __forceinline__ auto operator()(const A& a, const B& b) const {
    return a < b ? a : b;
  }
};

struct Maximum {
  template <typename A, typename B>
  __host__ __device__ __forceinline__ auto operator()(const A& a, const B& b) const {
    return a > b ? a : b;
  }
};

/// ``j -> xf(base[j * stride])``: the whole non-contiguous read as one functor. ``stride`` is a
/// runtime value because the lift may only know it as a symbol (TSVC ``s318``'s ``inc``).
template <typename T, typename Xf>
struct StridedGather {
  const T* base;
  long long stride;
  Xf xf;
  using result_type = T;  // pre-C++11-invoke_result thrust looks this up rather than deducing it
  __host__ __device__ __forceinline__ T operator()(long long j) const { return xf(base[j * stride]); }
};

/// Random-access iterator over an index functor (:struct:`StridedGather`, :struct:`IndexedPair`) with an explicit
/// iterator category; rocPRIM's ArgMax requires ``random_access_iterator_tag``, which thrust-backed iterators do not
/// report.
template <typename F>
struct GatherIterator {
  using T = typename F::result_type;
  using iterator_category = ::std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = long long;
  using pointer = void;
  using reference = T;  // a computed element has no storage to refer to

  F gather;
  difference_type index;

  __host__ __device__ __forceinline__ T operator*() const { return gather(index); }
  __host__ __device__ __forceinline__ T operator[](difference_type n) const { return gather(index + n); }
  __host__ __device__ __forceinline__ GatherIterator operator+(difference_type n) const {
    return GatherIterator{gather, index + n};
  }
  __host__ __device__ __forceinline__ GatherIterator operator-(difference_type n) const {
    return GatherIterator{gather, index - n};
  }
  __host__ __device__ __forceinline__ difference_type operator-(const GatherIterator& o) const {
    return index - o.index;
  }
  __host__ __device__ __forceinline__ GatherIterator& operator+=(difference_type n) {
    index += n;
    return *this;
  }
  __host__ __device__ __forceinline__ GatherIterator& operator-=(difference_type n) {
    index -= n;
    return *this;
  }
  __host__ __device__ __forceinline__ GatherIterator& operator++() {
    ++index;
    return *this;
  }
  __host__ __device__ __forceinline__ GatherIterator operator++(int) {
    GatherIterator prev = *this;
    ++index;
    return prev;
  }
  __host__ __device__ __forceinline__ GatherIterator& operator--() {
    --index;
    return *this;
  }
  __host__ __device__ __forceinline__ GatherIterator operator--(int) {
    GatherIterator prev = *this;
    --index;
    return prev;
  }
  __host__ __device__ __forceinline__ bool operator==(const GatherIterator& o) const { return index == o.index; }
  __host__ __device__ __forceinline__ bool operator!=(const GatherIterator& o) const { return index != o.index; }
  __host__ __device__ __forceinline__ bool operator<(const GatherIterator& o) const { return index < o.index; }
  __host__ __device__ __forceinline__ bool operator>(const GatherIterator& o) const { return index > o.index; }
  __host__ __device__ __forceinline__ bool operator<=(const GatherIterator& o) const { return index <= o.index; }
  __host__ __device__ __forceinline__ bool operator>=(const GatherIterator& o) const { return index >= o.index; }
};

template <typename F>
__host__ __device__ __forceinline__ GatherIterator<F> operator+(long long n, const GatherIterator<F>& it) {
  return it + n;
}

/// The iterator CUB reduces over when the operand is strided and/or transformed. ``Xf`` is named
/// explicitly by the caller; ``T`` is deduced from ``base``.
template <typename Xf, typename T>
inline GatherIterator<StridedGather<T, Xf>> gather_iterator(const T* base, long long stride) {
  return GatherIterator<StridedGather<T, Xf>>{StridedGather<T, Xf>{base, stride, Xf{}}, 0};
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
  template <typename T>
  using Buf = ArgBuf<T>;
  template <typename InIt, typename T>
  static gpuError_t call(void* scratch, size_t& needed, InIt in, ArgBuf<T>* out, long long items, gpuStream_t stream) {
#if DACE_CUB_ARG_REDUCE_SPLIT_OUTPUTS
    return ::gpucub::DeviceReduce::ArgMax(scratch, needed, in, out ? &out->value : nullptr, out ? &out->key : nullptr,
                                          items, stream);
#else
    return ::gpucub::DeviceReduce::ArgMax(scratch, needed, in, out, (int)items, stream);
#endif
  }
};

struct ArgMinOp {
  template <typename T>
  using Buf = ArgBuf<T>;
  template <typename InIt, typename T>
  static gpuError_t call(void* scratch, size_t& needed, InIt in, ArgBuf<T>* out, long long items, gpuStream_t stream) {
#if DACE_CUB_ARG_REDUCE_SPLIT_OUTPUTS
    return ::gpucub::DeviceReduce::ArgMin(scratch, needed, in, out ? &out->value : nullptr, out ? &out->key : nullptr,
                                          items, stream);
#else
    return ::gpucub::DeviceReduce::ArgMin(scratch, needed, in, out, (int)items, stream);
#endif
  }
};

/// ``j -> (in[j], j)``, the pairs :struct:`ArgPairOp` folds. A NaN past position 0 becomes the neutral
/// pair: the loop never takes one, and the fold would otherwise read it as the NaN seed it keeps.
template <typename InIt, typename T>
struct IndexedPair {
  InIt in;
  ::dace::arg_pair<T, long long> neutral;
  using result_type = ::dace::arg_pair<T, long long>;
  __host__ __device__ __forceinline__ result_type operator()(long long j) const {
    const T v = in[j];
    return (j == 0 || v == v) ? result_type{v, j} : neutral;
  }
};

/// An arg-reduction that folds (value, index) pairs by ``DeviceReduce::Reduce`` with the ``dace/arg_reduction.h``
/// combiner: ``Last`` is a non-strict (``>=`` / ``<=``) guard, ties to the HIGHER index, which CUB's ArgMax / ArgMin
/// cannot say.
template <bool Max, bool Last>
struct ArgPairOp {
  template <typename T>
  using Buf = ::dace::arg_pair<T, long long>;
  struct Combine {
    template <typename P>
    __host__ __device__ __forceinline__ P operator()(const P& out, const P& in) const {
      return ::dace::arg_combine<Max, Last>(out, in);
    }
  };
  template <typename InIt, typename T>
  static gpuError_t call(void* scratch, size_t& needed, InIt in, Buf<T>* out, long long items, gpuStream_t stream) {
    Buf<T> neutral;
    ::dace::arg_seed(neutral, Max, Last);
    const GatherIterator<IndexedPair<InIt, T>> pairs{IndexedPair<InIt, T>{in, neutral}, 0};
    return ::gpucub::DeviceReduce::Reduce(scratch, needed, pairs, out, items, Combine{}, neutral, stream);
  }
};

template <typename T>
inline void arg_answer(const ArgBuf<T>& got, T* val_out, long long* idx_out) {
  if (val_out != nullptr) *val_out = got.value;
  *idx_out = (long long)got.key;
}
template <typename T>
inline void arg_answer(const ::dace::arg_pair<T, long long>& got, T* val_out, long long* idx_out) {
  if (val_out != nullptr) *val_out = got.__ar_v;
  *idx_out = got.__ar_i;
}

/// Arg-reduction over a device sequence, answered on the host. ``val_out`` may be null; ties break toward
/// the lower index, or the higher one under :struct:`ArgPairOp`.
template <typename Op, typename InIt, typename T>
inline gpuError_t arg_reduce(InIt in, T* val_out, long long* idx_out, long long items, gpuStream_t stream) {
  using Buf = typename Op::template Buf<T>;
  size_t needed = 0;
  gpuError_t status = Op::call(nullptr, needed, in, (Buf*)nullptr, items, stream);
  if (status != gpuSuccess) return status;
  const size_t head = ((sizeof(Buf) + 255) / 256) * 256;
  void* scratch = get_scratch<ReduceTag>(head + needed, stream, &status);
  if (scratch == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;
  Buf* dev = (Buf*)scratch;
  status = Op::call((char*)scratch + head, needed, in, dev, items, stream);
  if (status != gpuSuccess) return status;
  Buf host;
  status = gpuMemcpyAsync(&host, dev, sizeof(Buf), gpuMemcpyDeviceToHost, stream);
  if (status != gpuSuccess) return status;
  status = gpuStreamSynchronize(stream);
  if (status != gpuSuccess) return status;
  arg_answer(host, val_out, idx_out);
  return gpuSuccess;
}

}  // namespace cub
}  // namespace dace

#endif  // __DACE_CUB_COMPAT_CUH
