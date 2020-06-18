#ifndef __DACE_REDUCTION_H
#define __DACE_REDUCTION_H

#include <cstdint>

#include "types.h"
#include "math.h"  // for ::min, ::max

#ifdef __CUDACC__
    #include "../../../external/cub/cub/device/device_segmented_reduce.cuh"
    #include "../../../external/cub/cub/device/device_reduce.cuh"
    #include "../../../external/cub/cub/block/block_reduce.cuh"
    #include "../../../external/cub/cub/iterator/counting_input_iterator.cuh"
    #include "../../../external/cub/cub/iterator/transform_input_iterator.cuh"
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

    // Specialization of CAS for float and double
    template <>
    struct wcr_custom<float> {
        template <typename WCR>
        static DACE_HDFI void reduce_atomic(WCR wcr, float *ptr, const float& value) {
            // The slowest kind of atomic operations (locked/compare-and-swap),
            // this should only happen in case of unrecognized lambdas
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
                // Adapted from CUDA's pre-v8.0 double atomicAdd implementation
                int *iptr = (int *)ptr;
                int old = *iptr, assumed;
                do {
                    assumed = old;
                    old = atomicCAS(iptr, assumed, 
                        __float_as_int(wcr(__int_as_float(assumed), value)));
                } while (assumed != old);
            #else
                #pragma omp critical
                *ptr = wcr(*ptr, value);
            #endif
        }

        // Non-conflicting version --> no critical section
        template <typename WCR>
        static DACE_HDFI void reduce(WCR wcr, float *ptr, const float& value) {
            *ptr = wcr(*ptr, value);
        }
    };

    template <>
    struct wcr_custom<double> {
        template <typename WCR>
        static DACE_HDFI void reduce_atomic(WCR wcr, double *ptr, const double& value) {
            // The slowest kind of atomic operations (locked/compare-and-swap),
            // this should only happen in case of unrecognized lambdas
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
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
            #else
                #pragma omp critical
                *ptr = wcr(*ptr, value);
            #endif
        }

        // Non-conflicting version --> no critical section
        template <typename WCR>
        static DACE_HDFI void reduce(WCR wcr, double *ptr, const double& value) {
            *ptr = wcr(*ptr, value);
        }
    };
    // End of specialization

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

// Implementation of double atomicAdd for CUDA architectures prior to 6.0
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    template <>
    struct _wcr_fixed<ReductionType::Sum, double> {
        static DACE_HDFI void reduce(double *ptr, const double& value) { *ptr += value; }

        static DACE_HDFI void reduce_atomic(double *ptr, const double& value) {
            unsigned long long int* address_as_ull = (unsigned long long int*)ptr;
            unsigned long long int old = *address_as_ull, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(value + __longlong_as_double(assumed)));
            } while (assumed != old);
        }

        DACE_HDFI double operator()(const double &a, const double &b) const { return a + b; }
    };
#endif

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
                wcr_custom<T>::reduce_atomic(
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
                wcr_custom<T>::reduce_atomic(
                    _wcr_fixed<ReductionType::Max, T>(), ptr, value);
            #endif
        }


        DACE_HDFI T operator()(const T &a, const T &b) const { return ::max(a, b); }
    };

    // Specialization for floating point types
    template <>
    struct _wcr_fixed<ReductionType::Min, float> {

        static DACE_HDFI void reduce(float *ptr, const float& value) { *ptr = ::min(*ptr, value); }
                

        static DACE_HDFI void reduce_atomic(float *ptr, const float& value) { 
            wcr_custom<float>::reduce_atomic(
                _wcr_fixed<ReductionType::Min, float>(), ptr, value);
        }


        DACE_HDFI float operator()(const float &a, const float &b) const { return ::min(a, b); }
    };
    
    template <>
    struct _wcr_fixed<ReductionType::Max, float> {

        static DACE_HDFI void reduce(float *ptr, const float& value) { *ptr = ::max(*ptr, value); }
                

        static DACE_HDFI void reduce_atomic(float *ptr, const float& value) { 
            wcr_custom<float>::reduce_atomic(
                _wcr_fixed<ReductionType::Max, float>(), ptr, value);
        }

        DACE_HDFI float operator()(const float &a, const float &b) const { return ::max(a, b); }
    };

    template <>
    struct _wcr_fixed<ReductionType::Min, double> {

        static DACE_HDFI void reduce(double *ptr, const double& value) { *ptr = ::min(*ptr, value); }
                

        static DACE_HDFI void reduce_atomic(double *ptr, const double& value) { 
            wcr_custom<double>::reduce_atomic(
                _wcr_fixed<ReductionType::Min, double>(), ptr, value);
        }


        DACE_HDFI double operator()(const double &a, const double &b) const { return ::min(a, b); }
    };
    
    template <>
    struct _wcr_fixed<ReductionType::Max, double> {

        static DACE_HDFI void reduce(double *ptr, const double& value) { *ptr = ::max(*ptr, value); }
                

        static DACE_HDFI void reduce_atomic(double *ptr, const double& value) { 
            wcr_custom<double>::reduce_atomic(
                _wcr_fixed<ReductionType::Max, double>(), ptr, value);
        }

        DACE_HDFI double operator()(const double &a, const double &b) const { return ::max(a, b); }
    };
    // End of specialization

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
#endif

}  // namespace dace


#endif  // __DACE_REDUCTION_H
