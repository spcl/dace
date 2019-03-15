#ifndef __DACE_REDUCTION_H
#define __DACE_REDUCTION_H

#include <cstdint>

#include "types.h"
#include "math.h"  // for ::min, ::max

#ifdef __CUDACC__
    #include "../../../external/cub/cub/device/device_reduce.cuh"
    #include "../../../external/cub/cub/block/block_reduce.cuh"
#endif

// Specializations for reductions implemented in frameworks like OpenMP, MPI

namespace dace {

    // Internal type. See below for wcr_fixed external type, which selects
    // the implementation according to T's properties.
    template <ReductionType REDTYPE, typename T>
    struct _wcr_fixed
    {
        static DACE_HDFI void reduce(T *ptr, const T& value);

        static DACE_HDFI void reduce_atomic(T *ptr, const T& value);

        DACE_HDFI T operator()(const T &a, const T &b) const;
    };


    // Custom reduction with a lambda function
    template <typename T>
    struct wcr_custom {
        template <typename WCR>
        static DACE_HDFI void reduce_atomic(WCR wcr, T *ptr, const T& value) {
            // The slowest kind of atomic operations (locked/compare-and-swap),
            // this should only happen in case of unrecognized lambdas
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
                // Adapted from CUDA's pre-v8.0 double atomicAdd implementation
                T old = *ptr, assumed;
                do {
                    assumed = old;
                    old = atomicCAS(ptr, assumed, wcr(assumed, value));
                } while (assumed != old);
            #else
                #pragma omp critical
                *ptr = wcr(*ptr, value);
            #endif
        }

        // Non-conflicting version --> no critical section
        template <typename WCR>
        static DACE_HDFI void reduce(WCR wcr, T *ptr, const T& value) {
            *ptr = wcr(*ptr, value);
        }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Sum, T> {
        static DACE_HDFI void reduce(T *ptr, const T& value) { *ptr += value; }
        
        static DACE_HDFI void reduce_atomic(T *ptr, const T& value) { 
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
                atomicAdd(ptr, value);
            #else
                #pragma omp atomic
                *ptr += value; 
            #endif
        }

        DACE_HDFI T operator()(const T &a, const T &b) const { return a + b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Product, T> {

        static DACE_HDFI void reduce(T *ptr, const T& value) { *ptr *= value; }
        

        static DACE_HDFI void reduce_atomic(T *ptr, const T& value) { 
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
                wcr_custom<T>::reduce(
                    _wcr_fixed<ReductionType::Product, T>(), ptr, value);
            #else
                #pragma omp atomic
                *ptr *= value; 
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a * b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Min, T> {

        static DACE_HDFI void reduce(T *ptr, const T& value) { *ptr = ::min(*ptr, value); }
                

        static DACE_HDFI void reduce_atomic(T *ptr, const T& value) { 
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
                atomicMin(ptr, value);
            #else
                wcr_custom<T>::reduce(
                    _wcr_fixed<ReductionType::Min, T>(), ptr, value);
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return ::min(a, b); }
    };
    
    template <typename T>
    struct _wcr_fixed<ReductionType::Max, T> {

        static DACE_HDFI void reduce(T *ptr, const T& value) { *ptr = ::max(*ptr, value); }
                

        static DACE_HDFI void reduce_atomic(T *ptr, const T& value) { 
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
                atomicMax(ptr, value);
            #else
              wcr_custom<T>::reduce(
                  _wcr_fixed<ReductionType::Max, T>(), ptr, value);
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return ::max(a, b); }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Logical_And, T> {

        static DACE_HDFI void reduce(T *ptr, const T& value) { *ptr = (*ptr && value); }
                

        static DACE_HDFI void reduce_atomic(T *ptr, const T& value) { 
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
                atomicAnd(ptr, value ? T(1) : T(0));
            #else
                T val = (value ? T(1) : T(0));
                #pragma omp atomic
                *ptr &= val;
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a && b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Bitwise_And, T> {

        static DACE_HDFI void reduce(T *ptr, const T& value) { *ptr &= value; }
                

        static DACE_HDFI void reduce_atomic(T *ptr, const T& value) { 
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
                atomicAnd(ptr, value);
            #else
                #pragma omp atomic
                *ptr &= value;
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a & b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Logical_Or, T> {

        static DACE_HDFI void reduce(T *ptr, const T& value) { *ptr = (*ptr || value); }
                

        static DACE_HDFI void reduce_atomic(T *ptr, const T& value) { 
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
                atomicOr(ptr, value ? T(1) : T(0));
            #else
                T val = (value ? T(1) : T(0));
                #pragma omp atomic
                *ptr |= val;
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a || b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Bitwise_Or, T> {

        static DACE_HDFI void reduce(T *ptr, const T& value) { *ptr |= value; }
                        

        static DACE_HDFI void reduce_atomic(T *ptr, const T& value) { 
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
                atomicOr(ptr, value);
            #else
                #pragma omp atomic
                *ptr |= value;
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a | b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Logical_Xor, T> {

        static DACE_HDFI void reduce(T *ptr, const T& value) { *ptr = (*ptr != value); }
                        

        static DACE_HDFI void reduce_atomic(T *ptr, const T& value) { 
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
                atomicXor(ptr, value ? T(1) : T(0));
            #else
                T val = (value ? T(1) : T(0));
                #pragma omp atomic
                *ptr ^= val;
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a != b; }
    };

    template <typename T>
    struct _wcr_fixed<ReductionType::Bitwise_Xor, T> {

        static DACE_HDFI void reduce(T *ptr, const T& value) { *ptr ^= value; }
                        

        static DACE_HDFI void reduce_atomic(T *ptr, const T& value) { 
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
                atomicXor(ptr, value);
            #else
                #pragma omp atomic
                *ptr ^= value;
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return a ^ b; }
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
        static DACE_HDFI void reduce(T *ptr, const T& value) 
        {
            _wcr_fixed<REDTYPE, T>::reduce(ptr, value);
        }

        static DACE_HDFI void reduce_atomic(T *ptr, const T& value) 
        {
            wcr_custom<T>::template reduce_atomic(
                _wcr_fixed<REDTYPE, T>(), ptr, value);
        }
    };

    // When atomics are supported, use _wcr_fixed normally
    template <ReductionType REDTYPE, typename T>
    struct wcr_fixed<REDTYPE, T, EnableIfScalar<T> >
    {
        static DACE_HDFI void reduce(T *ptr, const T& value)
        {
            _wcr_fixed<REDTYPE, T>::reduce(ptr, value);
        }

        static DACE_HDFI void reduce_atomic(T *ptr, const T& value)
        {
            _wcr_fixed<REDTYPE, T>::reduce_atomic(ptr, value);
        }

        DACE_HDFI T operator()(const T &a, const T &b) const
        {
            return _wcr_fixed<REDTYPE, T>()(a, b);
        }
    };

}  // namespace dace


#endif  // __DACE_REDUCTION_H
