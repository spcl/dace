// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_REDUCTION_H
#define __DACE_REDUCTION_H

#include <algorithm>
#include <cstdint>

// ``types.h`` first: ``math.h`` pulls in ``ITE.h`` before defining ``DACE_CONSTEXPR``,
// so this header is only self-contained when the macros are already in scope.
#include "types.h"

#include "math.h"  // for ::min, ::max
#include "vector.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
// Which CUB this is, and whether it is reachable at all, is answered once in gpucub.cuh. Asking for
// <cub/cub.cuh> here instead selected the vendored NVIDIA copy on every HIP build, where the header
// is not visible -- and that copy needs cuda.h.
#include "cuda/gpucub.cuh"
// The cub iterators are deprecated in favour of thrust's, and warn from CCCL 2.8 on (CUDA 12.8).
// rocThrust ships the same two, so this is not a CUDA-only preference.
#if __has_include(<thrust/iterator/counting_iterator.h>)
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#define DACE_THRUST_ITERATORS
#endif
#endif

// Not __HIPCC__: hip_common.h defines it in the host pass too on the NVIDIA platform.
#ifdef __HIP_DEVICE_COMPILE__
// HIP supports the same set of atomic ops as CUDA SM 6.0+
#define DACE_USE_GPU_ATOMICS
#define DACE_USE_GPU_DOUBLE_ATOMICS
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
#define DACE_USE_GPU_ATOMICS
#if __CUDA_ARCH__ >= 600
#define DACE_USE_GPU_DOUBLE_ATOMICS
#endif
#endif

// Specializations for reductions implemented in frameworks like OpenMP, MPI

namespace dace {

// Internal type. See below for wcr_fixed external type, which selects
// the implementation according to T's properties.
template <ReductionType REDTYPE, typename T>
struct _wcr_fixed {
  static DACE_HDFI T reduce_atomic(T *ptr, const T &value);

  DACE_HDFI T operator()(const T &a, const T &b) const;
};

// Custom reduction with a lambda function
template <typename T>
struct wcr_custom {
  template <typename WCR>
  static DACE_HDFI T reduce_atomic(WCR wcr, T *ptr, const T &value) {
    // The slowest kind of atomic operations (locked/compare-and-swap),
    // this should only happen in case of unrecognized lambdas
    T old;
#ifdef DACE_USE_GPU_ATOMICS
    // Adapted from CUDA's pre-v8.0 double atomicAdd implementation
    T assumed;
    old = *ptr;
    do {
      assumed = old;
      old = atomicCAS(ptr, assumed, wcr(assumed, value));
    } while (assumed != old);
#else
#pragma omp critical
    {
      old = *ptr;
      *ptr = wcr(old, value);
    }
#endif

    return old;
  }

  // Non-conflicting version --> no critical section
  template <typename WCR>
  static DACE_HDFI T reduce(WCR wcr, T *ptr, const T &value) {
    T old = *ptr;
    *ptr = wcr(old, value);
    return old;
  }
};

// Specialization of CAS for float and double
template <>
struct wcr_custom<float> {
  template <typename WCR>
  static DACE_HDFI float reduce_atomic(WCR wcr, float *ptr,
                                       const float &value) {
// The slowest kind of atomic operations (locked/compare-and-swap),
// this should only happen in case of unrecognized lambdas
#ifdef DACE_USE_GPU_ATOMICS
    // Adapted from CUDA's pre-v8.0 double atomicAdd implementation
    int *iptr = (int *)ptr;
    int old = *iptr, assumed;
    do {
      assumed = old;
      old = atomicCAS(iptr, assumed,
                      __float_as_int(wcr(__int_as_float(assumed), value)));
    } while (assumed != old);
    return __int_as_float(old);
#else
    float old;
#pragma omp critical
    {
      old = *ptr;
      *ptr = wcr(old, value);
    }
    return old;
#endif
  }

  // Non-conflicting version --> no critical section
  template <typename WCR>
  static DACE_HDFI float reduce(WCR wcr, float *ptr, const float &value) {
    float old = *ptr;
    *ptr = wcr(old, value);
    return old;
  }
};

template <>
struct wcr_custom<double> {
  template <typename WCR>
  static DACE_HDFI double reduce_atomic(WCR wcr, double *ptr,
                                        const double &value) {
// The slowest kind of atomic operations (locked/compare-and-swap),
// this should only happen in case of unrecognized lambdas
#ifdef DACE_USE_GPU_ATOMICS
    // Adapted from CUDA's pre-v8.0 double atomicAdd implementation
    unsigned long long *iptr = (unsigned long long *)ptr;
    unsigned long long old = *ptr, assumed;
    do {
      assumed = old;
      old = atomicCAS(
          iptr, assumed,
          __double_as_longlong(wcr(__longlong_as_double(assumed), value)));
    } while (assumed != old);
    return __longlong_as_double(old);
#else
    double old;
#pragma omp critical
    {
      old = *ptr;
      *ptr = wcr(old, value);
    }
    return old;
#endif
  }

  // Non-conflicting version --> no critical section
  template <typename WCR>
  static DACE_HDFI double reduce(WCR wcr, double *ptr, const double &value) {
    double old = *ptr;
    *ptr = wcr(old, value);
    return old;
  }
};
// End of specialization

template <typename T>
struct _wcr_fixed<ReductionType::Sum, T> {
  static DACE_HDFI T reduce_atomic(T *ptr, const T &value) {
#ifdef DACE_USE_GPU_ATOMICS
    return atomicAdd(ptr, value);
#elif defined(_OPENMP) && _OPENMP >= 201107
    T old;
#pragma omp atomic capture
    {
      old = *ptr;
      *ptr += value;
    }
    return old;
#else
#pragma omp atomic
    *ptr += value;
    return T(0);  // Unsupported
#endif
  }

  DACE_HDFI T operator()(const T &a, const T &b) const { return a + b; }
};

// Implementation of double atomicAdd for CUDA architectures prior to 6.0
#if defined(DACE_USE_GPU_ATOMICS) && !defined(DACE_USE_GPU_DOUBLE_ATOMICS)
template <>
struct _wcr_fixed<ReductionType::Sum, double> {
  static DACE_HDFI double reduce_atomic(double *ptr, const double &value) {
    unsigned long long int *address_as_ull = (unsigned long long int *)ptr;
    unsigned long long int old = *address_as_ull, assumed;
    do {
      assumed = old;
      old = atomicCAS(
          address_as_ull, assumed,
          __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }

  DACE_HDFI double operator()(const double &a, const double &b) const {
    return a + b;
  }
};
#endif

#if defined(DACE_USE_GPU_ATOMICS)
template <>
struct _wcr_fixed<ReductionType::Sum, int64_t> {
  static DACE_HDFI int64_t reduce_atomic(int64_t *ptr, const int64_t &value) {
    return _wcr_fixed<ReductionType::Sum, unsigned long long>::reduce_atomic(
        (unsigned long long *)ptr, static_cast<unsigned long long>(value));
  }

  DACE_HDFI int64_t operator()(const int64_t &a, const int64_t &b) const {
    return a + b;
  }
};

template <>
struct _wcr_fixed<ReductionType::Sum, uint64_t> {
  static DACE_HDFI uint64_t reduce_atomic(uint64_t *ptr,
                                          const uint64_t &value) {
    return _wcr_fixed<ReductionType::Sum, unsigned long long>::reduce_atomic(
        (unsigned long long *)ptr, static_cast<unsigned long long>(value));
  }

  DACE_HDFI uint64_t operator()(const uint64_t &a, const uint64_t &b) const {
    return a + b;
  }
};
#endif

template <typename T>
struct _wcr_fixed<ReductionType::Product, T> {
  static DACE_HDFI T reduce_atomic(T *ptr, const T &value) {
#ifdef DACE_USE_GPU_ATOMICS
    return wcr_custom<T>::reduce(_wcr_fixed<ReductionType::Product, T>(), ptr,
                                 value);
#elif defined(_OPENMP) && _OPENMP >= 201107
    T old;
#pragma omp atomic capture
    {
      old = *ptr;
      *ptr *= value;
    }
    return old;
#else
#pragma omp atomic
    *ptr *= value;
    return T(0);  // Unsupported
#endif
  }

  DACE_HDFI T operator()(const T &a, const T &b) const { return a * b; }
};

template <typename T>
struct _wcr_fixed<ReductionType::Min, T> {
  static DACE_HDFI T reduce_atomic(T *ptr, const T &value) {
#ifdef DACE_USE_GPU_ATOMICS
    return atomicMin(ptr, value);
#else
    return wcr_custom<T>::reduce_atomic(_wcr_fixed<ReductionType::Min, T>(),
                                        ptr, value);
#endif
  }

  DACE_HDFI T operator()(const T &a, const T &b) const { return ::min(a, b); }
};

template <typename T>
struct _wcr_fixed<ReductionType::Max, T> {
  static DACE_HDFI T reduce_atomic(T *ptr, const T &value) {
#ifdef DACE_USE_GPU_ATOMICS
    return atomicMax(ptr, value);
#else
    return wcr_custom<T>::reduce_atomic(_wcr_fixed<ReductionType::Max, T>(),
                                        ptr, value);
#endif
  }

  DACE_HDFI T operator()(const T &a, const T &b) const { return ::max(a, b); }
};

// Specialization for floating point types
template <>
struct _wcr_fixed<ReductionType::Min, float> {
  static DACE_HDFI float reduce_atomic(float *ptr, const float &value) {
    return wcr_custom<float>::reduce_atomic(
        _wcr_fixed<ReductionType::Min, float>(), ptr, value);
  }

  DACE_HDFI float operator()(const float &a, const float &b) const {
    return ::min(a, b);
  }
};

template <>
struct _wcr_fixed<ReductionType::Max, float> {
  static DACE_HDFI float reduce_atomic(float *ptr, const float &value) {
    return wcr_custom<float>::reduce_atomic(
        _wcr_fixed<ReductionType::Max, float>(), ptr, value);
  }

  DACE_HDFI float operator()(const float &a, const float &b) const {
    return ::max(a, b);
  }
};

template <>
struct _wcr_fixed<ReductionType::Min, double> {
  static DACE_HDFI double reduce_atomic(double *ptr, const double &value) {
    return wcr_custom<double>::reduce_atomic(
        _wcr_fixed<ReductionType::Min, double>(), ptr, value);
  }

  DACE_HDFI double operator()(const double &a, const double &b) const {
    return ::min(a, b);
  }
};

template <>
struct _wcr_fixed<ReductionType::Max, double> {
  static DACE_HDFI double reduce_atomic(double *ptr, const double &value) {
    return wcr_custom<double>::reduce_atomic(
        _wcr_fixed<ReductionType::Max, double>(), ptr, value);
  }

  DACE_HDFI double operator()(const double &a, const double &b) const {
    return ::max(a, b);
  }
};

// half has no CUDA atomicMin/Max overload → 16-bit CAS (smallest atomic word, sm_70+),
// mirroring the float/double CAS specializations. Host + pre-sm_70: promote to float
// (half min/max exact through float → bit-identical either way).
template <>
struct _wcr_fixed<ReductionType::Min, half> {
  static DACE_HDFI half reduce_atomic(half *ptr, const half &value) {
#if defined(DACE_USE_GPU_ATOMICS) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    unsigned short int *iptr = (unsigned short int *)ptr;
    unsigned short int old = *iptr, assumed;
    do {
      assumed = old;
      old = atomicCAS(iptr, assumed,
                      __half_as_ushort(half(::min(float(__ushort_as_half(assumed)), float(value)))));
    } while (assumed != old);
    return __ushort_as_half(old);
#else
    half old = *ptr;
    *ptr = half(::min(float(old), float(value)));
    return old;
#endif
  }

  DACE_HDFI half operator()(const half &a, const half &b) const {
    return half(::min(float(a), float(b)));
  }
};

template <>
struct _wcr_fixed<ReductionType::Max, half> {
  static DACE_HDFI half reduce_atomic(half *ptr, const half &value) {
#if defined(DACE_USE_GPU_ATOMICS) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    unsigned short int *iptr = (unsigned short int *)ptr;
    unsigned short int old = *iptr, assumed;
    do {
      assumed = old;
      old = atomicCAS(iptr, assumed,
                      __half_as_ushort(half(::max(float(__ushort_as_half(assumed)), float(value)))));
    } while (assumed != old);
    return __ushort_as_half(old);
#else
    half old = *ptr;
    *ptr = half(::max(float(old), float(value)));
    return old;
#endif
  }

  DACE_HDFI half operator()(const half &a, const half &b) const {
    return half(::max(float(a), float(b)));
  }
};
// End of specialization

template <typename T>
struct _wcr_fixed<ReductionType::Logical_And, T> {
  static DACE_HDFI T reduce_atomic(T *ptr, const T &value) {
#ifdef DACE_USE_GPU_ATOMICS
    return atomicAnd(ptr, value ? T(1) : T(0));
#elif defined(_OPENMP) && _OPENMP >= 201107
    T old;
    T val = (value ? T(1) : T(0));
#pragma omp atomic capture
    {
      old = *ptr;
      *ptr &= val;
    }
    return old;
#else
    T val = (value ? T(1) : T(0));
#pragma omp atomic
    *ptr &= val;
    return T(0);  // Unsupported
#endif
  }

  DACE_HDFI T operator()(const T &a, const T &b) const { return a && b; }
};

template <typename T>
struct _wcr_fixed<ReductionType::Bitwise_And, T> {
  static DACE_HDFI T reduce_atomic(T *ptr, const T &value) {
#ifdef DACE_USE_GPU_ATOMICS
    return atomicAnd(ptr, value);
#elif defined(_OPENMP) && _OPENMP >= 201107
    T old;
#pragma omp atomic capture
    {
      old = *ptr;
      *ptr &= value;
    }
    return old;
#else
#pragma omp atomic
    *ptr &= value;
    return T(0);  // Unsupported
#endif
  }

  DACE_HDFI T operator()(const T &a, const T &b) const { return a & b; }
};

template <typename T>
struct _wcr_fixed<ReductionType::Logical_Or, T> {
  static DACE_HDFI T reduce_atomic(T *ptr, const T &value) {
#ifdef DACE_USE_GPU_ATOMICS
    return atomicOr(ptr, value ? T(1) : T(0));
#elif defined(_OPENMP) && _OPENMP >= 201107
    T old;
    T val = (value ? T(1) : T(0));
#pragma omp atomic capture
    {
      old = *ptr;
      *ptr |= val;
    }
    return old;
#else
    T val = (value ? T(1) : T(0));
#pragma omp atomic
    *ptr |= val;
    return T(0);  // Unsupported
#endif
  }

  DACE_HDFI T operator()(const T &a, const T &b) const { return a || b; }
};

template <typename T>
struct _wcr_fixed<ReductionType::Bitwise_Or, T> {
  static DACE_HDFI T reduce_atomic(T *ptr, const T &value) {
#ifdef DACE_USE_GPU_ATOMICS
    return atomicOr(ptr, value);
#elif defined(_OPENMP) && _OPENMP >= 201107
    T old;
#pragma omp atomic capture
    {
      old = *ptr;
      *ptr |= value;
    }
    return old;
#else
#pragma omp atomic
    *ptr |= value;
    return T(0);  // Unsupported
#endif
  }

  DACE_HDFI T operator()(const T &a, const T &b) const { return a | b; }
};

template <typename T>
struct _wcr_fixed<ReductionType::Logical_Xor, T> {
  static DACE_HDFI T reduce_atomic(T *ptr, const T &value) {
#ifdef DACE_USE_GPU_ATOMICS
    return atomicXor(ptr, value ? T(1) : T(0));
#elif defined(_OPENMP) && _OPENMP >= 201107
    T old;
    T val = (value ? T(1) : T(0));
#pragma omp atomic capture
    {
      old = *ptr;
      *ptr ^= val;
    }
    return old;
#else
    T val = (value ? T(1) : T(0));
#pragma omp atomic
    *ptr ^= val;
    return T(0);  // Unsupported
#endif
  }

  DACE_HDFI T operator()(const T &a, const T &b) const { return a != b; }
};

template <typename T>
struct _wcr_fixed<ReductionType::Bitwise_Xor, T> {
  static DACE_HDFI T reduce_atomic(T *ptr, const T &value) {
#ifdef DACE_USE_GPU_ATOMICS
    return atomicXor(ptr, value);
#elif defined(_OPENMP) && _OPENMP >= 201107
    T old;
#pragma omp atomic capture
    {
      old = *ptr;
      *ptr ^= value;
    }
    return old;
#else
#pragma omp atomic
    *ptr ^= value;
    return T(0);  // Unsupported
#endif
  }

  DACE_HDFI T operator()(const T &a, const T &b) const { return a ^ b; }
};

template <typename T>
struct _wcr_fixed<ReductionType::Exchange, T> {
  static DACE_HDFI T reduce_atomic(T *ptr, const T &value) {
#ifdef DACE_USE_GPU_ATOMICS
    return atomicExch(ptr, value);
#else
    T old;
#pragma omp critical
    {
      old = *ptr;
      *ptr = value;
    }
    return old;
#endif
  }

  DACE_HDFI T operator()(const T &, const T &b) const { return b; }
};

//////////////////////////////////////////////////////////////////////////

// Specialization that regresses to critical section / locked update for
// unsupported types
template <typename T>
using EnableIfScalar = typename std::enable_if<std::is_scalar<T>::value>::type;

// Any vector type that is not of length 1, or struct/complex types
// do not support atomics. In these cases, we regress to locked updates.
template <ReductionType REDTYPE, typename T, typename SFINAE = void>
struct wcr_fixed {
  static DACE_HDFI T reduce(T *ptr, const T &value) {
    T old = *ptr;
    *ptr = _wcr_fixed<REDTYPE, T>()(old, value);
    return old;
  }

  static DACE_HDFI T reduce_atomic(T *ptr, const T &value) {
    return wcr_custom<T>::template reduce_atomic<
        decltype(_wcr_fixed<REDTYPE, T>())>(_wcr_fixed<REDTYPE, T>(), ptr,
                                            value);
  }
};

// When atomics are supported, use _wcr_fixed normally
template <ReductionType REDTYPE, typename T>
struct wcr_fixed<REDTYPE, T, EnableIfScalar<T> > {
  static DACE_HDFI T reduce(T *ptr, const T &value) {
    T old = *ptr;
    *ptr = _wcr_fixed<REDTYPE, T>()(old, value);
    return old;
  }

  static DACE_HDFI T reduce_atomic(T *ptr, const T &value) {
    return _wcr_fixed<REDTYPE, T>::reduce_atomic(ptr, value);
  }

  DACE_HDFI T operator()(const T &a, const T &b) const {
    return _wcr_fixed<REDTYPE, T>()(a, b);
  }

  // Vector -> Scalar versions
  template <int N>
  static DACE_HDFI T vreduce(T *ptr, const dace::vec<T, N> &value) {
    T old = *ptr;

    T scal = value[0];
    __DACE_UNROLL
    for (int i = 1; i < N; ++i) scal = _wcr_fixed<REDTYPE, T>()(scal, value[i]);

    *ptr = _wcr_fixed<REDTYPE, T>()(old, scal);
    return old;
  }

  template <int N>
  static DACE_HDFI T vreduce_atomic(T *ptr, const dace::vec<T, N> &value) {
    T scal = value[0];
    __DACE_UNROLL
    for (int i = 1; i < N; ++i) scal = _wcr_fixed<REDTYPE, T>()(scal, value[i]);

    return _wcr_fixed<REDTYPE, T>::reduce_atomic(ptr, scal);
  }
};

//////////////////////////////////////////////////////////////////////////
// dace::reduce -- the CPU lowering of the ``Reduce`` library node.
//
// Two shapes, mirroring ``dace/scan.hpp``'s blocked three-phase scan: ``dace::reduce::<op>`` is
// the PARALLEL one (an OpenMP ``reduction`` clause), ``dace::reduce::seq::<op>`` the SEQUENTIAL
// one (naked loop, no pragma).
// CONTRACT: the parallel entries carry no runtime nesting check -- ``ExpandReduceOpenMP`` picks the
// shape statically from the node's SCOPE. An external caller already inside its own ``omp`` region
// gets OpenMP's nested default, a one-thread team: correct, slower.
//
// Every entry point folds ``n`` elements from ``in``, ``s`` apart, into ``seed`` and RETURNS the
// result. ``seed`` carries the identity (or the output element's previous contents), so ``n <= 0``
// returns it untouched and no op needs an identity of its own -- which is what makes ``min``/``max``
// expressible. The accumulator type is ``seed``'s, i.e. the OUTPUT dtype, so an integer array reduced
// into a real accumulates in the real type. The index space is walked directly; ``s == 1`` branches
// only to keep the contiguous case unit-stride for the vectorizer.
//
// SEMANTICS GUARANTEED. FP association: the parallel form reassociates (the team splits the range,
// the runtime combines partials), so a floating-point result MOVES WITH ``OMP_NUM_THREADS``, while
// the sequential form associates the same way at every thread count and is bit-reproducible. That
// fixed association is PAIRWISE for ``sum`` and ``product`` over a rounding accumulator -- a binary
// tree over blocks of ``detail::PAIRWISE_BLOCK``, the shape numpy's own reduction uses -- because a
// left-to-right fold stops moving once the accumulator outgrows the addend, and a long single
// precision sum then answers with a visibly wrong number. Reproducible, not left-to-right: code
// that needs a specific summation order has to spell it out itself. min/max and NaN: both
// forms compare with ``std::min``/``std::max``, ``(b < a) ? b : a`` with the accumulator on the left,
// so a NaN in the DATA never wins and the result is the min/max over the non-NaN elements at every
// thread count; a NaN ``seed`` propagates through the SEQUENTIAL form but goes through the runtime's
// own combiner in the parallel one and is unspecified there. That differs from ``_wcr_fixed<Min>``
// above, whose ``::min`` lets a NaN through. Integer/bitwise/logical ops are exact and thread-count
// independent.

namespace reduce {

// THE SUPPORTED ELEMENT TYPES, in one place. Each has an OpenMP reduction: built in, declared below
// (complex), or declared in ``types.h`` (the low-precision structs, whose combiner promotes to
// ``float`` and rounds back on store). No other class type: no operator detection, no fallback, just
// the rejection in ``detail::checked_seed``.
template <typename T>
struct is_reducible : std::integral_constant<bool, std::is_arithmetic<T>::value> {};
template <>
struct is_reducible<complex64> : std::true_type {};
template <>
struct is_reducible<complex128> : std::true_type {};
template <>
struct is_reducible<float16> : std::true_type {};
template <>
struct is_reducible<bfloat16> : std::true_type {};
template <>
struct is_reducible<float8_e4m3fn> : std::true_type {};
template <>
struct is_reducible<float8_e5m2> : std::true_type {};

// The reducible set MINUS the complex types, which have neither an ordering nor a boolean
// conversion: ``min`` / ``max`` and ``&&`` / ``||`` are defined on exactly this set. Only ``+`` and
// ``*`` are defined on a complex operand, matching the OpenMP reductions declared below.
template <typename T>
struct is_real : is_reducible<T> {};
template <>
struct is_real<complex64> : std::false_type {};
template <>
struct is_real<complex128> : std::false_type {};

// ``&`` / ``|`` / ``^`` exist for an integral operand only, in C++ and in OpenMP alike.
template <typename T>
struct is_bitwise : std::integral_constant<bool, std::is_integral<T>::value> {};

#ifdef _OPENMP
// Complex needs a user-defined reduction (OpenMP has none for a class type), and only ``+`` / ``*``.
#pragma omp declare reduction(+ : complex64 : omp_out += omp_in) initializer(omp_priv = complex64(0))
#pragma omp declare reduction(+ : complex128 : omp_out += omp_in) initializer(omp_priv = complex128(0))
#pragma omp declare reduction(* : complex64 : omp_out *= omp_in) initializer(omp_priv = complex64(1))
#pragma omp declare reduction(* : complex128 : omp_out *= omp_in) initializer(omp_priv = complex128(1))
#endif

namespace detail {

/// Returns ``seed``; the single place an unsupported element type is refused. Every entry point
/// runs its seed through one of these three before touching the data, so an out-of-policy element
/// type is a NAMED static assertion rather than whatever the op's operator noise happens to be.
template <typename T, typename U>
inline T checked_seed(T seed) {
  static_assert(is_reducible<T>::value && is_reducible<U>::value,
                "dace::reduce: unsupported element type. Supported: builtin arithmetic types, "
                "dace::complex64 / complex128, and dace::float16 / bfloat16 / float8_e4m3fn / float8_e5m2.");
  return seed;
}

/// ``min`` / ``max`` and the logical ops: the reducible set minus complex.
template <typename T, typename U>
inline T real_seed(T seed) {
  static_assert(is_real<T>::value && is_real<U>::value,
                "dace::reduce: min/max and the logical reductions need a real element type. "
                "dace::complex64 / complex128 have no ordering and no boolean conversion -- only "
                "sum and product are defined for them.");
  return checked_seed<T, U>(seed);
}

/// ``&`` / ``|`` / ``^``: integral operands only.
template <typename T, typename U>
inline T bitwise_seed(T seed) {
  static_assert(is_bitwise<T>::value && is_bitwise<U>::value,
                "dace::reduce: the bitwise reductions need an integral element type. No bitwise "
                "operator exists for a floating-point or complex operand.");
  return checked_seed<T, U>(seed);
}

/// The element, as an operand of the accumulator's type. Two complex types of DIFFERENT width do
/// not combine -- ``std::complex<double> + std::complex<float>`` matches no ``operator+`` -- so that
/// pair alone is widened here. Every other in-policy pair combines on its own and is passed through
/// untouched, which keeps a narrow accumulator over wider input rounding exactly where it did.
template <typename T, typename U>
inline auto operand(const U &x) {
  if constexpr (is_reducible<T>::value && !is_real<T>::value && !is_real<U>::value && !std::is_same<T, U>::value)
    return T(x);
  else
    return x;
}

/// Left-to-right fold of ``n`` elements of ``in`` taken ``s`` apart, seeded with ``seed``. The
/// caller has already run ``seed`` through its op's policy check.
template <typename T, typename U, typename Op>
inline T fold(const U *in, long n, long s, T seed, Op op) {
  T acc = seed;
  if (s == 1) {
    for (long i = 0; i < n; ++i) acc = op(acc, in[i]);
  } else {
    for (long i = 0; i < n; ++i) acc = op(acc, in[i * s]);
  }
  return acc;
}

/// Elements below which the pairwise walk stops splitting and folds the block through the lanes
/// below. numpy's own block size, and for the same reason: large enough that the recursion is
/// amortised over real work, small enough that the block's error stays under the tree's.
constexpr long PAIRWISE_BLOCK = 128;

/// The bottom of the pairwise walk: ``n >= 1`` elements of ``in`` taken ``s`` apart, folded through
/// eight independent chains and combined as a tree.
///
/// Eight because the additions within one chain are then the only ones that wait on each other, so
/// the block runs at the adder's throughput rather than its latency -- and because eight partials
/// each grow eight times slower than one running total would, which is the accuracy half of the
/// same choice. numpy's reduction has this shape.
template <typename T, typename U, typename Op, typename Merge>
inline T fold_block(const U *in, long n, long s, Op op, Merge merge) {
  if (n < 8) {
    T acc = static_cast<T>(operand<T, U>(in[0]));
    for (long i = 1; i < n; ++i) acc = op(acc, in[i * s]);
    return acc;
  }
  T r0 = static_cast<T>(operand<T, U>(in[0]));
  T r1 = static_cast<T>(operand<T, U>(in[s]));
  T r2 = static_cast<T>(operand<T, U>(in[2 * s]));
  T r3 = static_cast<T>(operand<T, U>(in[3 * s]));
  T r4 = static_cast<T>(operand<T, U>(in[4 * s]));
  T r5 = static_cast<T>(operand<T, U>(in[5 * s]));
  T r6 = static_cast<T>(operand<T, U>(in[6 * s]));
  T r7 = static_cast<T>(operand<T, U>(in[7 * s]));
  long i = 8;
  for (; i + 7 < n; i += 8) {
    r0 = op(r0, in[i * s]);
    r1 = op(r1, in[(i + 1) * s]);
    r2 = op(r2, in[(i + 2) * s]);
    r3 = op(r3, in[(i + 3) * s]);
    r4 = op(r4, in[(i + 4) * s]);
    r5 = op(r5, in[(i + 5) * s]);
    r6 = op(r6, in[(i + 6) * s]);
    r7 = op(r7, in[(i + 7) * s]);
  }
  for (; i < n; ++i) r0 = op(r0, in[i * s]);
  return merge(merge(merge(r0, r1), merge(r2, r3)), merge(merge(r4, r5), merge(r6, r7)));
}

/// Pairwise (binary tree) fold of ``n >= 1`` elements of ``in`` taken ``s`` apart, with no seed --
/// the leftmost element IS the seed, so no identity is needed and the caller's seed is merged once
/// at the top. Sequential and deterministic like ``fold``, but the rounding error of a floating
/// point accumulation grows as O(log n) rather than O(n): a left-to-right fold stops moving as soon
/// as the accumulator outgrows the addend, and a tree keeps the partials the same size as each
/// other. ``op`` combines the accumulator with an element, ``merge`` two accumulators.
template <typename T, typename U, typename Op, typename Merge>
inline T pairwise(const U *in, long n, long s, Op op, Merge merge) {
  if (n <= PAIRWISE_BLOCK) return fold_block<T, U>(in, n, s, op, merge);
  const long half = n / 2;
  return merge(pairwise<T, U>(in, half, s, op, merge), pairwise<T, U>(in + half * s, n - half, s, op, merge));
}

/// ``pairwise`` for an accumulator that rounds, ``fold`` for one that does not.
///
/// An integral accumulator reassociates exactly, so the tree would buy it nothing and cost it the
/// recursion. Everything else -- float, double, complex, the low-precision structs -- is where the
/// association is the whole question.
template <typename T, typename U, typename Op, typename Merge>
inline T associative_fold(const U *in, long n, long s, T seed, Op op, Merge merge) {
  if constexpr (std::is_integral<T>::value) {
    return fold(in, n, s, seed, op);
  } else {
    if (n <= 0) return seed;
    return merge(seed, pairwise<T, U>(in, n, s, op, merge));
  }
}

}  // namespace detail

// --- PARALLEL SHAPE ------------------------------------------------------------

template <typename T, typename U>
inline T sum(const U *in, long n, long s, T seed) {
  T acc = detail::checked_seed<T, U>(seed);
  if (s == 1) {
#pragma omp parallel for simd reduction(+ : acc)
    for (long i = 0; i < n; ++i) acc = static_cast<T>(acc + detail::operand<T, U>(in[i]));
  } else {
#pragma omp parallel for reduction(+ : acc)
    for (long i = 0; i < n; ++i) acc = static_cast<T>(acc + detail::operand<T, U>(in[i * s]));
  }
  return acc;
}

template <typename T, typename U>
inline T product(const U *in, long n, long s, T seed) {
  T acc = detail::checked_seed<T, U>(seed);
  if (s == 1) {
#pragma omp parallel for simd reduction(* : acc)
    for (long i = 0; i < n; ++i) acc = static_cast<T>(acc * detail::operand<T, U>(in[i]));
  } else {
#pragma omp parallel for reduction(* : acc)
    for (long i = 0; i < n; ++i) acc = static_cast<T>(acc * detail::operand<T, U>(in[i * s]));
  }
  return acc;
}

// min/max carry no ``simd``: no vectorizer folds the NaN-swallowing ``(b < a) ? b : a`` without
// reassociating comparisons, and clang warns about the transformation it refused. Measured free.
// The ``reduction(op : acc)`` clause and the bare operator below are ILL-FORMED for a rejected T
// (no ``operator<``/``&``/``&&`` for a class type such as complex) -- and on this compiler, a hard
// error raised from THAT statement suppresses the ``static_assert`` above rather than joining it, so
// the user sees the pragma's generic complaint instead of the runtime's named one. ``if constexpr``
// keeps the policy-violating statement out of the instantiation entirely: the assert is then the
// only diagnostic. Guard, don't restructure -- the taken branch is untouched for every legal T.
template <typename T, typename U>
inline T min(const U *in, long n, long s, T seed) {
  T acc = detail::real_seed<T, U>(seed);
  if constexpr (is_real<T>::value && is_real<U>::value) {
    if (s == 1) {
#pragma omp parallel for reduction(min : acc)
      for (long i = 0; i < n; ++i) acc = std::min<T>(acc, static_cast<T>(in[i]));
    } else {
#pragma omp parallel for reduction(min : acc)
      for (long i = 0; i < n; ++i) acc = std::min<T>(acc, static_cast<T>(in[i * s]));
    }
  }
  return acc;
}

template <typename T, typename U>
inline T max(const U *in, long n, long s, T seed) {
  T acc = detail::real_seed<T, U>(seed);
  if constexpr (is_real<T>::value && is_real<U>::value) {
    if (s == 1) {
#pragma omp parallel for reduction(max : acc)
      for (long i = 0; i < n; ++i) acc = std::max<T>(acc, static_cast<T>(in[i]));
    } else {
#pragma omp parallel for reduction(max : acc)
      for (long i = 0; i < n; ++i) acc = std::max<T>(acc, static_cast<T>(in[i * s]));
    }
  }
  return acc;
}

// Names its own reduction: the built-in ``&`` initializes its private copy from an unsigned all-ones
// literal that clang reports as a signedness change on signed ``T``. Same identity, spelled -1.
template <typename T, typename U>
inline T bitwise_and(const U *in, long n, long s, T seed) {
  T acc = detail::bitwise_seed<T, U>(seed);
  if constexpr (is_bitwise<T>::value && is_bitwise<U>::value) {
#pragma omp declare reduction(dace_band : T : omp_out = static_cast<T>(omp_out &omp_in)) initializer(omp_priv = static_cast<T>(-1))
    if (s == 1) {
#pragma omp parallel for simd reduction(dace_band : acc)
      for (long i = 0; i < n; ++i) acc = static_cast<T>(acc & in[i]);
    } else {
#pragma omp parallel for reduction(dace_band : acc)
      for (long i = 0; i < n; ++i) acc = static_cast<T>(acc & in[i * s]);
    }
  }
  return acc;
}

template <typename T, typename U>
inline T bitwise_or(const U *in, long n, long s, T seed) {
  T acc = detail::bitwise_seed<T, U>(seed);
  if constexpr (is_bitwise<T>::value && is_bitwise<U>::value) {
    if (s == 1) {
#pragma omp parallel for simd reduction(| : acc)
      for (long i = 0; i < n; ++i) acc = static_cast<T>(acc | in[i]);
    } else {
#pragma omp parallel for reduction(| : acc)
      for (long i = 0; i < n; ++i) acc = static_cast<T>(acc | in[i * s]);
    }
  }
  return acc;
}

template <typename T, typename U>
inline T bitwise_xor(const U *in, long n, long s, T seed) {
  T acc = detail::bitwise_seed<T, U>(seed);
  if constexpr (is_bitwise<T>::value && is_bitwise<U>::value) {
    if (s == 1) {
#pragma omp parallel for simd reduction(^ : acc)
      for (long i = 0; i < n; ++i) acc = static_cast<T>(acc ^ in[i]);
    } else {
#pragma omp parallel for reduction(^ : acc)
      for (long i = 0; i < n; ++i) acc = static_cast<T>(acc ^ in[i * s]);
    }
  }
  return acc;
}

template <typename T, typename U>
inline T logical_and(const U *in, long n, long s, T seed) {
  T acc = detail::real_seed<T, U>(seed);
  if constexpr (is_real<T>::value && is_real<U>::value) {
    if (s == 1) {
#pragma omp parallel for simd reduction(&& : acc)
      for (long i = 0; i < n; ++i) acc = static_cast<T>(acc && in[i]);
    } else {
#pragma omp parallel for reduction(&& : acc)
      for (long i = 0; i < n; ++i) acc = static_cast<T>(acc && in[i * s]);
    }
  }
  return acc;
}

template <typename T, typename U>
inline T logical_or(const U *in, long n, long s, T seed) {
  T acc = detail::real_seed<T, U>(seed);
  if constexpr (is_real<T>::value && is_real<U>::value) {
    if (s == 1) {
#pragma omp parallel for simd reduction(|| : acc)
      for (long i = 0; i < n; ++i) acc = static_cast<T>(acc || in[i]);
    } else {
#pragma omp parallel for reduction(|| : acc)
      for (long i = 0; i < n; ++i) acc = static_cast<T>(acc || in[i * s]);
    }
  }
  return acc;
}

// --- SEQUENTIAL SHAPE ----------------------------------------------------------

namespace seq {

template <typename T, typename U>
inline T sum(const U *in, long n, long s, T seed) {
  return detail::associative_fold(
      in, n, s, detail::checked_seed<T, U>(seed),
      [](T a, U b) { return static_cast<T>(a + detail::operand<T, U>(b)); },
      [](T a, T b) { return static_cast<T>(a + b); });
}

template <typename T, typename U>
inline T product(const U *in, long n, long s, T seed) {
  return detail::associative_fold(
      in, n, s, detail::checked_seed<T, U>(seed),
      [](T a, U b) { return static_cast<T>(a * detail::operand<T, U>(b)); },
      [](T a, T b) { return static_cast<T>(a * b); });
}

// Same ``if constexpr`` guard as the parallel shape above: the lambda body below is ill-formed for
// a rejected T (no ``&``/``&&`` for a class type), and instantiating it alongside the ``static_assert``
// in ``real_seed``/``bitwise_seed`` lets the compiler's own operator error mask the runtime's named
// one. Keeping the fold out of the instantiation for a policy-violating T leaves only the assert.
template <typename T, typename U>
inline T min(const U *in, long n, long s, T seed) {
  T acc = detail::real_seed<T, U>(seed);
  if constexpr (is_real<T>::value && is_real<U>::value)
    acc = detail::fold(in, n, s, acc, [](T a, U b) { return std::min<T>(a, static_cast<T>(b)); });
  return acc;
}

template <typename T, typename U>
inline T max(const U *in, long n, long s, T seed) {
  T acc = detail::real_seed<T, U>(seed);
  if constexpr (is_real<T>::value && is_real<U>::value)
    acc = detail::fold(in, n, s, acc, [](T a, U b) { return std::max<T>(a, static_cast<T>(b)); });
  return acc;
}

template <typename T, typename U>
inline T bitwise_and(const U *in, long n, long s, T seed) {
  T acc = detail::bitwise_seed<T, U>(seed);
  if constexpr (is_bitwise<T>::value && is_bitwise<U>::value)
    acc = detail::fold(in, n, s, acc, [](T a, U b) { return static_cast<T>(a & b); });
  return acc;
}

template <typename T, typename U>
inline T bitwise_or(const U *in, long n, long s, T seed) {
  T acc = detail::bitwise_seed<T, U>(seed);
  if constexpr (is_bitwise<T>::value && is_bitwise<U>::value)
    acc = detail::fold(in, n, s, acc, [](T a, U b) { return static_cast<T>(a | b); });
  return acc;
}

template <typename T, typename U>
inline T bitwise_xor(const U *in, long n, long s, T seed) {
  T acc = detail::bitwise_seed<T, U>(seed);
  if constexpr (is_bitwise<T>::value && is_bitwise<U>::value)
    acc = detail::fold(in, n, s, acc, [](T a, U b) { return static_cast<T>(a ^ b); });
  return acc;
}

template <typename T, typename U>
inline T logical_and(const U *in, long n, long s, T seed) {
  T acc = detail::real_seed<T, U>(seed);
  if constexpr (is_real<T>::value && is_real<U>::value)
    acc = detail::fold(in, n, s, acc, [](T a, U b) { return static_cast<T>(a && b); });
  return acc;
}

template <typename T, typename U>
inline T logical_or(const U *in, long n, long s, T seed) {
  T acc = detail::real_seed<T, U>(seed);
  if constexpr (is_real<T>::value && is_real<U>::value)
    acc = detail::fold(in, n, s, acc, [](T a, U b) { return static_cast<T>(a || b); });
  return acc;
}

}  // namespace seq
}  // namespace reduce

// Not CUDA-only: ``Reduce``'s device expansion emits ``dace::stridedIterator`` for a strided
// segmented reduce, and that expansion serves both backends.
#if defined(__CUDACC__) || defined(__HIPCC__)
struct StridedIteratorHelper {
  explicit StridedIteratorHelper(size_t stride) : stride(stride) {}
  size_t stride;

  __host__ __device__ __forceinline__ size_t
  operator()(const size_t &index) const {
    return index * stride;
  }
};

inline auto stridedIterator(size_t stride) {
#ifdef DACE_THRUST_ITERATORS
  thrust::counting_iterator
#else
  gpucub::CountingInputIterator
#endif
      <int>
          counting_iterator(0);
  StridedIteratorHelper conversion_op(stride);
#ifdef DACE_THRUST_ITERATORS
  thrust::transform_iterator<decltype(conversion_op),
                             decltype(counting_iterator)>
      itr(counting_iterator, conversion_op);
#else
  gpucub::TransformInputIterator<int, decltype(conversion_op),
                              decltype(counting_iterator)>
      itr(counting_iterator, conversion_op);
#endif

  return itr;
}
#endif

#if defined(__CUDACC__)
template <ReductionType REDTYPE, typename T>
struct warpReduce {
  static DACE_DFI T reduce(T v) {
    for (int i = 1; i < 32; i = i * 2)
      v = _wcr_fixed<REDTYPE, T>()(v, __shfl_xor_sync(0xffffffff, v, i));
    return v;
  }

  template <int NUM_MW>
  static DACE_DFI T mini_reduce(T v) {
    for (int i = 1; i < NUM_MW; i = i * 2)
      v = _wcr_fixed<REDTYPE, T>()(v, __shfl_xor_sync(0xffffffff, v, i));
    return v;
  }
};
#elif defined(__HIPCC__)
template <ReductionType REDTYPE, typename T>
struct warpReduce {
  static DACE_DFI T reduce(T v) {
    for (int i = 1; i < warpSize; i = i * 2)
      v = _wcr_fixed<REDTYPE, T>()(v, __shfl_xor(v, i));
    return v;
  }

  template <int NUM_MW>
  static DACE_DFI T mini_reduce(T v) {
    for (int i = 1; i < NUM_MW; i = i * 2)
      v = _wcr_fixed<REDTYPE, T>()(v, __shfl_xor(v, i));
    return v;
  }
};
#endif

}  // namespace dace

#endif  // __DACE_REDUCTION_H
