// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_REDUCTION_H
#define __DACE_REDUCTION_H

#include <cstdint>

#include "types.h"
#include "vector.h"
#include "math.h"  // for ::min, ::max

#ifdef __CUDACC__
    #include "../../../external/cub/cub/device/device_segmented_reduce.cuh"
    #include "../../../external/cub/cub/device/device_reduce.cuh"
    #include "../../../external/cub/cub/block/block_reduce.cuh"
    #include "../../../external/cub/cub/iterator/counting_input_iterator.cuh"
    #include "../../../external/cub/cub/iterator/transform_input_iterator.cuh"
#endif

#ifdef __HIPCC__
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
    struct _wcr_fixed
    {
        static DACE_HDFI T reduce_atomic(T *ptr, const T& value);

        DACE_HDFI T operator()(const T &a, const T &b) const;
    };


    // Custom reduction with a lambda function
    template <typename T>
    struct wcr_custom {
        template <typename WCR>
        static DACE_HDFI T reduce_atomic(WCR wcr, T *ptr, const T& value) {
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
        static DACE_HDFI T reduce(WCR wcr, T *ptr, const T& value) {
            T old = *ptr;
            *ptr = wcr(old, value);
            return old;
        }
    };

    // Specialization of CAS for float and double
    template <>
    struct wcr_custom<float> {
        template <typename WCR>
        static DACE_HDFI float reduce_atomic(WCR wcr, float *ptr, const float& value) {
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
        static DACE_HDFI float reduce(WCR wcr, float *ptr, const float& value) {
            float old = *ptr;
            *ptr = wcr(old, value);
            return old;
        }
    };

    template <>
    struct wcr_custom<double> {
        template <typename WCR>
        static DACE_HDFI double reduce_atomic(WCR wcr, double *ptr, const double& value) {
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
                        __double_as_longlong(
                                wcr(__longlong_as_double(assumed),
                                    value)));
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
        static DACE_HDFI double reduce(WCR wcr, double *ptr, const double& value) {
            double old = *ptr;
            *ptr = wcr(old, value);
            return old;
        }
    };
    // End of specialization

    template <typename T>
    struct _wcr_fixed<ReductionType::Sum, T> {
       
        static DACE_HDFI T reduce_atomic(T *ptr, const T& value) {
            #ifdef DACE_USE_GPU_ATOMICS
                return atomicAdd(ptr, value);
            #elif defined (_OPENMP) && _OPENMP >= 201107
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
                return T(0); // Unsupported
            #endif
        }

        DACE_HDFI T operator()(const T &a, const T &b) const { return a + b; }
    };

// Implementation of double atomicAdd for CUDA architectures prior to 6.0
#if defined(DACE_USE_GPU_ATOMICS) && !defined(DACE_USE_GPU_DOUBLE_ATOMICS)
    template <>
    struct _wcr_fixed<ReductionType::Sum, double> {

        static DACE_HDFI double reduce_atomic(double *ptr, const double& value) {
            unsigned long long int* address_as_ull = (unsigned long long int*)ptr;
            unsigned long long int old = *address_as_ull, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(value + __longlong_as_double(assumed)));
            } while (assumed != old);
            return __longlong_as_double(old);
        }

        DACE_HDFI double operator()(const double &a, const double &b) const { return a + b; }
    };
#endif

#if defined(DACE_USE_GPU_ATOMICS)
    template <>
    struct _wcr_fixed<ReductionType::Sum, long long> {
       
        static DACE_HDFI long long reduce_atomic(long long *ptr, const long long& value) {
            return _wcr_fixed<ReductionType::Sum, unsigned long long>::reduce_atomic((
                unsigned long long *)ptr, 
                static_cast<unsigned long long>(value));
        }

        DACE_HDFI long long operator()(const long long &a, const long long &b) const { return a + b; }
    };
#endif

    template <typename T>
    struct _wcr_fixed<ReductionType::Product, T> {

        static DACE_HDFI T reduce_atomic(T *ptr, const T& value) { 
            #ifdef DACE_USE_GPU_ATOMICS
                return wcr_custom<T>::reduce(
                    _wcr_fixed<ReductionType::Product, T>(), ptr, value);
            #elif defined (_OPENMP) && _OPENMP >= 201107
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
                return T(0); // Unsupported
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a * b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Min, T> {

        static DACE_HDFI T reduce_atomic(T *ptr, const T& value) { 
            #ifdef DACE_USE_GPU_ATOMICS
                return atomicMin(ptr, value);
            #else
                return wcr_custom<T>::reduce_atomic(
                    _wcr_fixed<ReductionType::Min, T>(), ptr, value);
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return ::min(a, b); }
    };
    
    template <typename T>
    struct _wcr_fixed<ReductionType::Max, T> {

        static DACE_HDFI T reduce_atomic(T *ptr, const T& value) { 
            #ifdef DACE_USE_GPU_ATOMICS
                return atomicMax(ptr, value);
            #else
                return wcr_custom<T>::reduce_atomic(
                    _wcr_fixed<ReductionType::Max, T>(), ptr, value);
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return ::max(a, b); }
    };

    // Specialization for floating point types
    template <>
    struct _wcr_fixed<ReductionType::Min, float> {

        static DACE_HDFI float reduce_atomic(float *ptr, const float& value) { 
            return wcr_custom<float>::reduce_atomic(
                _wcr_fixed<ReductionType::Min, float>(), ptr, value);
        }


        DACE_HDFI float operator()(const float &a, const float &b) const { return ::min(a, b); }
    };
    
    template <>
    struct _wcr_fixed<ReductionType::Max, float> {

        static DACE_HDFI float reduce_atomic(float *ptr, const float& value) { 
            return wcr_custom<float>::reduce_atomic(
                _wcr_fixed<ReductionType::Max, float>(), ptr, value);
        }

        DACE_HDFI float operator()(const float &a, const float &b) const { return ::max(a, b); }
    };

    template <>
    struct _wcr_fixed<ReductionType::Min, double> {

        static DACE_HDFI double reduce_atomic(double *ptr, const double& value) { 
            return wcr_custom<double>::reduce_atomic(
                _wcr_fixed<ReductionType::Min, double>(), ptr, value);
        }


        DACE_HDFI double operator()(const double &a, const double &b) const { return ::min(a, b); }
    };
    
    template <>
    struct _wcr_fixed<ReductionType::Max, double> {

        static DACE_HDFI double reduce_atomic(double *ptr, const double& value) { 
            return wcr_custom<double>::reduce_atomic(
                _wcr_fixed<ReductionType::Max, double>(), ptr, value);
        }

        DACE_HDFI double operator()(const double &a, const double &b) const { return ::max(a, b); }
    };
    // End of specialization

    template <typename T>
    struct _wcr_fixed<ReductionType::Logical_And, T> {

        static DACE_HDFI T reduce_atomic(T *ptr, const T& value) { 
            #ifdef DACE_USE_GPU_ATOMICS
                return atomicAnd(ptr, value ? T(1) : T(0));
            #elif defined (_OPENMP) && _OPENMP >= 201107
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
                return T(0); // Unsupported
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a && b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Bitwise_And, T> {

        static DACE_HDFI T reduce_atomic(T *ptr, const T& value) { 
            #ifdef DACE_USE_GPU_ATOMICS
                return atomicAnd(ptr, value);
            #elif defined (_OPENMP) && _OPENMP >= 201107
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
                return T(0); // Unsupported
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a & b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Logical_Or, T> {

        static DACE_HDFI T reduce_atomic(T *ptr, const T& value) { 
            #ifdef DACE_USE_GPU_ATOMICS
                return atomicOr(ptr, value ? T(1) : T(0));
            #elif defined (_OPENMP) && _OPENMP >= 201107
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
                return T(0); // Unsupported
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a || b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Bitwise_Or, T> {

        static DACE_HDFI T reduce_atomic(T *ptr, const T& value) { 
            #ifdef DACE_USE_GPU_ATOMICS
                return atomicOr(ptr, value);
            #elif defined (_OPENMP) && _OPENMP >= 201107
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
                return T(0); // Unsupported
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a | b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Logical_Xor, T> {

        static DACE_HDFI T reduce_atomic(T *ptr, const T& value) { 
            #ifdef DACE_USE_GPU_ATOMICS
                return atomicXor(ptr, value ? T(1) : T(0));
            #elif defined (_OPENMP) && _OPENMP >= 201107
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
                return T(0); // Unsupported
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a != b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Bitwise_Xor, T> {

        static DACE_HDFI T reduce_atomic(T *ptr, const T& value) { 
            #ifdef DACE_USE_GPU_ATOMICS
                return atomicXor(ptr, value);
            #elif defined (_OPENMP) && _OPENMP >= 201107
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
                return T(0); // Unsupported
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a ^ b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Exchange, T> {

        static DACE_HDFI T reduce_atomic(T *ptr, const T& value) { 
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

        DACE_HDFI T operator()(const T &a, const T &b) const { return b; }
    };

    //////////////////////////////////////////////////////////////////////////

    // Specialization that regresses to critical section / locked update for
    // unsupported types
    template<typename T>
    using EnableIfScalar = typename std::enable_if<std::is_scalar<T>::value>::type;

    // Any vector type that is not of length 1, or struct/complex types 
    // do not support atomics. In these cases, we regress to locked updates.
    template <ReductionType REDTYPE, typename T, typename SFINAE = void>
    struct wcr_fixed
    {
        static DACE_HDFI T reduce(T *ptr, const T& value)
        {
            T old = *ptr;
            *ptr = _wcr_fixed<REDTYPE, T>()(old, value);
            return old;
        }

        static DACE_HDFI T reduce_atomic(T *ptr, const T& value) 
        {
            return wcr_custom<T>::template reduce_atomic(
                _wcr_fixed<REDTYPE, T>(), ptr, value);
        }
    };

    // When atomics are supported, use _wcr_fixed normally
    template <ReductionType REDTYPE, typename T>
    struct wcr_fixed<REDTYPE, T, EnableIfScalar<T> >
    {
        static DACE_HDFI T reduce(T *ptr, const T& value)
        {
            T old = *ptr;
            *ptr = _wcr_fixed<REDTYPE, T>()(old, value);
            return old;
        }

        static DACE_HDFI T reduce_atomic(T *ptr, const T& value)
        {
            return _wcr_fixed<REDTYPE, T>::reduce_atomic(ptr, value);
        }

        DACE_HDFI T operator()(const T &a, const T &b) const
        {
            return _wcr_fixed<REDTYPE, T>()(a, b);
        }

        // Vector -> Scalar versions
        template <int N>
        static DACE_HDFI T vreduce(T *ptr, const dace::vec<T, N>& value)
        {
            T old = *ptr;

            T scal = value[0];
            __DACE_UNROLL
            for (int i = 1; i < N; ++i)
              scal = _wcr_fixed<REDTYPE, T>()(scal, value[i]);

            *ptr = _wcr_fixed<REDTYPE, T>()(old, scal);
            return old;
        }

        template <int N>
        static DACE_HDFI T vreduce_atomic(T *ptr, const dace::vec<T, N>& value)
        {
            T scal = value[0];
            __DACE_UNROLL
            for (int i = 1; i < N; ++i)
              scal = _wcr_fixed<REDTYPE, T>()(scal, value[i]);
            
            return _wcr_fixed<REDTYPE, T>::reduce_atomic(ptr, scal);
        }
    };


#ifdef __CUDACC__
    struct StridedIteratorHelper {
	explicit StridedIteratorHelper(size_t stride)
	    : stride(stride) {}
	size_t stride;

	__host__ __device__ __forceinline__
	size_t operator()(const size_t &index) const {
	    return index * stride;
	}
    };

    inline auto stridedIterator(size_t stride) {
        cub::CountingInputIterator<int> counting_iterator(0);
        StridedIteratorHelper conversion_op(stride);
        cub::TransformInputIterator<int, decltype(conversion_op), decltype(counting_iterator)> itr(counting_iterator, conversion_op);
        return itr;
    }

    template <ReductionType REDTYPE, typename T>
    struct warpReduce {
        static DACE_DFI T reduce(T v)
        {
            for (int i = 1; i < 32; i = i * 2)
                v = _wcr_fixed<REDTYPE, T>()(v, __shfl_xor_sync(0xffffffff, v, i));
            return v;
        }

        template<int NUM_MW>
        static DACE_DFI T mini_reduce(T v)
        {
            for (int i = 1; i < NUM_MW; i = i * 2)
                v = _wcr_fixed<REDTYPE, T>()(v, __shfl_xor_sync(0xffffffff, v, i));
            return v;
        }
    };
#endif

}  // namespace dace


#endif  // __DACE_REDUCTION_H
