// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_VECTOR_H
#define __DACE_VECTOR_H

#ifdef DACE_XILINX_DEVICE_CODE
#include "xilinx/vec.h"
#else // Don't include this file if building for Xilinx

#include "types.h"

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
        typedef T aligned;
        typedef T unaligned;
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

        #define DEFINE_VECTYPE(T, BASE_SIZE, N)                             \
        template<>                                                          \
        struct _vtype<T, N>                                                 \
        {                                                                   \
            typedef simplevec<T, N> aligned;                                \
            typedef aligned unaligned;                                      \
        };                                                                  

    #else
        #define DEFINE_VECTYPE(T, BASE_SIZE, N)                             \
        template<>                                                          \
        struct _vtype<T, N>                                                 \
        {                                                                   \
            typedef T aligned __attribute__((vector_size(N * BASE_SIZE)));  \
            typedef aligned __attribute__((aligned(BASE_SIZE))) unaligned;  \
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
    struct vector_type
    {
        typedef typename _vtype<T, N>::aligned aligned;
        typedef typename _vtype<T, N>::unaligned unaligned;
        typedef T element_type;
        static constexpr unsigned int size = N;
        typedef union {
            aligned v;
            T s[N];
        } access_aligned;
        typedef union {
            unaligned v;
            T s[N];
        } access_unaligned;
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
