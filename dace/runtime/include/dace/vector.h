// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_VECTOR_H
#define __DACE_VECTOR_H

#ifdef DACE_XILINX
#include "xilinx/vec.h"
#else // Don't include this file if building for Xilinx

#include "types.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
    #include "cuda/halfvec.cuh"
#endif

namespace dace
{
    //////////////////////////////////////////////////////////////////
    // Workaround for clang++
    // Defining all vector sizes at compile time
    template <typename T, int N>
    struct _vtype;

    // Identity type
    template <typename T>
    struct _vtype<T, 1>
    {
        using aligned = T;
        using unaligned = T;
    };

#if defined(__CUDACC__) || defined(__HIPCC__)
    // NOTE: This file is inline and MUST be included here
    #include "cuda/vectype.cuh"
#else
    #if defined(_MSC_VER)
        template <typename T, int N>
        struct simplevec;
        template <typename T>
        struct simplevec<T, 1> {
            union { struct { T x; }; T s[1]; };
            inline T operator[](int ind) const { return s[ind]; }
            template <typename U>
            inline simplevec<T, 1> operator*(const U& other) const
            {
                simplevec<T, 1> result;
                result.x = x * other;
                return result;
            }
            template <typename U>
            inline simplevec<T, 1> operator+(const U& other) const
            {
                simplevec<T, 1> result;
                result.x = x + other;
                return result;
            }
        };
        template <typename T>
        struct simplevec<T, 2>
        {
            union { struct { T x, y; }; T s[2]; };
            inline T operator[](int ind) const { return s[ind]; }
            template <typename U>
            inline simplevec<T, 2> operator*(const U& other) const
            {
                simplevec<T, 2> result;
                result.x = x * other;
                result.y = y * other;
                return result;
            }
            inline simplevec<T, 2> operator+(const simplevec<T, 2>& other) const
            {
                simplevec<T, 2> result;
                result.x = x + other.x;
                result.y = y + other.y;
                return result;
            }
        };
        template <typename T>
        struct simplevec<T, 3>
        {
            union { struct { T x, y, z; }; T s[3]; };
            inline T operator[](int ind) const { return s[ind]; }
            template <typename U>
            inline simplevec<T, 3> operator*(const U& other) const
            {
                simplevec<T, 3> result;
                for (int i = 0; i < 3; ++i) result.s[i] = s[i] * other;
                return result;
            }
            inline simplevec<T, 3> operator+(const simplevec<T, 3>& other) const
            {
                simplevec<T, 3> result;
                for (int i = 0; i < 3; ++i) result.s[i] = s[i] + other.s[i];
                return result;
            }
        };
        template <typename T>
        struct simplevec<T, 4>
        {
            union { struct { T x, y, z, w; }; T s[4]; };
            inline T operator[](int ind) const { return s[ind]; }
            template <typename U>
            inline simplevec<T, 4> operator*(const U& other) const
            {
                simplevec<T, 4> result;
                for (int i = 0; i < 4; ++i) result.s[i] = s[i] * other;
                return result;
            }
            inline simplevec<T, 4> operator+(const simplevec<T, 4>& other) const
            {
                simplevec<T, 4> result;
                for (int i = 0; i < 4; ++i) result.s[i] = s[i] + other.s[i];
                return result;
            }
        };
        template <typename T>
        struct simplevec<T, 8>
        {
            union { struct { T s0, s1, s2, s3, s4, s5, s6, s7; }; T s[8]; };
            inline T operator[](int ind) const { return s[ind]; }
            template <typename U>
            inline simplevec<T, 8> operator*(const U& other) const
            {
                simplevec<T, 8> result;
                for (int i = 0; i < 8; ++i) result.s[i] = s[i] * other;
                return result;
            }
            inline simplevec<T, 8> operator+(const simplevec<T, 8>& other) const
            {
                simplevec<T, 8> result;
                for (int i = 0; i < 8; ++i) result.s[i] = s[i] + other.s[i];
                return result;
            }
        };
        template <typename T>
        struct simplevec<T, 16>
        {
            union { 
                struct {
                    T s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
                      s14, s15;
                }; 
                T s[16]; 
            };
            inline T operator[](int ind) const { return s[ind]; }
            template <typename U>
            inline simplevec<T, 16> operator*(const U& other) const
            {
                simplevec<T, 16> result;
                for (int i = 0; i < 16; ++i) result.s[i] = s[i] * other;
                return result;
            }
            inline simplevec<T, 16> operator+(const simplevec<T, 16>& other) const
            {
                simplevec<T, 16> result;
                for (int i = 0; i < 16; ++i) result.s[i] = s[i] + other.s[i];
                return result;
            }
        };
        template <typename T>
        struct simplevec<T, 32>
        {
            union {
                struct {
                    T s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
                      s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25,
                      s26, s27, s28, s29, s30, s31;
                };
                T s[32];
            };
            inline T operator[](int ind) const { return s[ind]; }
            template <typename U>
            inline simplevec<T, 32> operator*(const U& other) const
            {
                simplevec<T, 32> result;
                for (int i = 0; i < 32; ++i) result.s[i] = s[i] * other;
                return result;
            }
            inline simplevec<T, 32> operator+(const simplevec<T, 32>& other) const
            {
                simplevec<T, 32> result;
                for (int i = 0; i < 32; ++i) result.s[i] = s[i] + other.s[i];
                return result;
            }
        };

        template <typename T, int N>
        static simplevec<T, N> operator+(const simplevec<T, N>& op1,
                                         const T& op2) {
            simplevec<T, N> result;
            for (int i = 0; i < N; ++i) result.s[i] = op1.s[i] + op2;
            return result;
        }
        template <typename T, int N>
        static simplevec<T, N> operator+(const T& op1,
                                         const simplevec<T, N>& op2) {
            simplevec<T, N> result;
            for (int i = 0; i < N; ++i) result.s[i] = op1 + op2.s[i];
            return result;
        }
        template <typename T, int N>
        static simplevec<T, N> operator*(const simplevec<T, N>& op1,
                                         const T& op2) {
            simplevec<T, N> result;
            for (int i = 0; i < N; ++i) result.s[i] = op1.s[i] * op2;
            return result;
        }
        template <typename T, int N>
        static simplevec<T, N> operator*(const T& op1,
                                         const simplevec<T, N>& op2) {
            simplevec<T, N> result;
            for (int i = 0; i < N; ++i) result.s[i] = op1 * op2.s[i];
            return result;
        }

        #define DEFINE_VECTYPE(T, BASE_SIZE, N)                             \
        template<>                                                          \
        struct _vtype<T, N>                                                 \
        {                                                                   \
            using aligned = simplevec<T, N>;                                \
            using unaligned = aligned;                                      \
        };                                                                  

    #else
        #define DEFINE_VECTYPE(T, BASE_SIZE, N)                             \
        template<>                                                          \
        struct _vtype<T, N>                                                 \
        {                                                                   \
            using aligned = T __attribute__((vector_size(N * BASE_SIZE)));  \
            using unaligned = aligned __attribute__((aligned(BASE_SIZE)));  \
        };
    #endif
        #define DEFINE_VECTYPE_ALLSIZES(T, BASE_SIZE)                       \
            DEFINE_VECTYPE(T, BASE_SIZE, 2);                                \
            DEFINE_VECTYPE(T, BASE_SIZE, 4);                                \
            DEFINE_VECTYPE(T, BASE_SIZE, 8);                                \
            DEFINE_VECTYPE(T, BASE_SIZE, 16);                               \
            DEFINE_VECTYPE(T, BASE_SIZE, 32);


        DEFINE_VECTYPE_ALLSIZES(int8   , 1);
        DEFINE_VECTYPE_ALLSIZES(int16  , 2);
        DEFINE_VECTYPE_ALLSIZES(int32  , 4);
        DEFINE_VECTYPE_ALLSIZES(int64  , 8);
        DEFINE_VECTYPE_ALLSIZES(uint8  , 1);
        DEFINE_VECTYPE_ALLSIZES(uint16 , 2);
        DEFINE_VECTYPE_ALLSIZES(uint32 , 4);
        DEFINE_VECTYPE_ALLSIZES(uint64 , 8);
    //  DEFINE_VECTYPE_ALLSIZES(float16, 2);
        DEFINE_VECTYPE_ALLSIZES(float32, 4);
        DEFINE_VECTYPE_ALLSIZES(float64, 8);
#endif    

    // END of workaround for clang++
    //////////////////////////////////////////////////////////////////

    template <typename T, unsigned int N>
    struct generalvec {
        T s[N];
        inline T const &operator[](size_t ind) const { return s[ind]; }
        inline T &operator[](size_t ind) { return s[ind]; }
    };

    namespace detail
    {
        // If the element type is not fundamental, for example when nesting
        // vector types within vector types, fall back on a generic 
        // implementation.

        template <typename T, unsigned int N, bool = std::is_fundamental<T>::value>
        struct simple_or_general;

        template <typename T, unsigned int N>
        struct simple_or_general<T, N, true> {
            using aligned = typename _vtype<T, N>::aligned;
            using unaligned = typename _vtype<T, N>::unaligned;
        };

        template <typename T, unsigned int N>
        struct simple_or_general<T, N, false> {
            using aligned = generalvec<T, N>;
            using unaligned = generalvec<T, N>;
        };

        template <typename T>
        struct simple_or_general<T, 1, true> {
            using aligned = T;
            using unaligned = T;
        };

        template <typename T>
        struct simple_or_general<T, 1, false> {
            using aligned = T;
            using unaligned = T;
        };

    }  // namespace detail

    template <typename T, unsigned int N>
    struct vector_type
    {
        using aligned = typename detail::simple_or_general<T, N>::aligned;
        using unaligned = typename detail::simple_or_general<T, N>::unaligned;
        using element_type = T;
        static constexpr unsigned int size = N;
        using access_aligned = union {
            aligned v;
            T s[N];
        };
        using access_unaligned = union {
            unaligned v;
            T s[N];
        };
    };

    template <typename T, unsigned int N>
    using vec = typename vector_type<T, N>::aligned;

    template <typename T, unsigned int N>
    using vecu = typename vector_type<T, N>::unaligned;

    template <typename T1, typename T2, unsigned int N>
    vec<T1, N> xtoy(vec<T2, N> x) {
        vec<T1, N> y;
        for (int i = 0; i < 4; ++i)
            y[i] = x[i];
        return y;
    }

}

#endif // XILINX_DEVICE_CODE
#endif  // __DACE_VECTOR_H
