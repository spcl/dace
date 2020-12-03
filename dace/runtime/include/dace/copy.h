// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_COPY_H
#define __DACE_COPY_H

#include "types.h"
#include "vector.h"

namespace dace
{
    template<typename T, typename U>
    inline void InitArray(T *ptr, const U& value, int size)
    {
        for (int i = 0; i < size; ++i)
            *ptr++ = T(value);
    }

    template <typename T, int VECLEN, int ALIGNED, int... COPYDIMS>
    struct CopyND;
    template <typename T, int VECLEN, int ALIGNED, int N>
    struct CopyNDDynamic;

    template <typename T, int VECLEN, int ALIGNED, int COPYDIM,
              int... OTHER_COPYDIMS>
    struct CopyND<T, VECLEN, ALIGNED, COPYDIM, OTHER_COPYDIMS...>
    {
        template <int SRC_STRIDE, int... OTHER_SRCDIMS>
        struct ConstSrc
        {
            template <typename... Args>
            static DACE_HDFI void Copy(const T *src, T *dst, const int& dst_stride, const Args&... dst_otherdims)
            {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
                // Memcpy specialization
                if (sizeof...(OTHER_COPYDIMS) == 0 && SRC_STRIDE == 1 && dst_stride == 1) {
                    memcpy(dst, src, COPYDIM * sizeof(T) * VECLEN);
                    return;
                }
#endif

                __DACE_UNROLL
                for (int i = 0; i < COPYDIM; ++i)
                    CopyND<T, VECLEN, ALIGNED, OTHER_COPYDIMS...>::template ConstSrc<OTHER_SRCDIMS...>::Copy(
                        src + i * SRC_STRIDE, dst + i * dst_stride, dst_otherdims...);
            }

            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc, const int& dst_stride, const Args&... dst_otherdims)
            {
                __DACE_UNROLL
                for (int i = 0; i < COPYDIM; ++i)
                    CopyND<T, VECLEN, ALIGNED, OTHER_COPYDIMS...>::template ConstSrc<OTHER_SRCDIMS...>::Accumulate(
                        src + i * SRC_STRIDE, dst + i * dst_stride, acc, dst_otherdims...);
            }
        };

        template <int DST_STRIDE, int... OTHER_DSTDIMS>
        struct ConstDst
        {
            template <typename... Args>
            static DACE_HDFI void Copy(const T *src, T *dst, const int& src_stride, const Args&... src_otherdims)
            {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
                // Memcpy specialization
                if (sizeof...(OTHER_COPYDIMS) == 0 && src_stride == 1 && DST_STRIDE == 1) {
                    memcpy(dst, src, COPYDIM * sizeof(T) * VECLEN);
                    return;
                }
#endif

                __DACE_UNROLL
                for (int i = 0; i < COPYDIM; ++i)
                    CopyND<T, VECLEN, ALIGNED, OTHER_COPYDIMS...>::template ConstDst<OTHER_DSTDIMS...>::Copy(
                        src + i * src_stride, dst + i * DST_STRIDE, src_otherdims...);
            }
            
            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc, const int& src_stride, const Args&... src_otherdims)
            {
                __DACE_UNROLL
                for (int i = 0; i < COPYDIM; ++i)
                    CopyND<T, VECLEN, ALIGNED, OTHER_COPYDIMS...>::template ConstDst<OTHER_DSTDIMS...>::Accumulate(
                        src + i * src_stride, dst + i * DST_STRIDE, acc, src_otherdims...);
            }
        };

        struct Dynamic
        {
            template <typename... Args>
            static DACE_HDFI void Copy(const T *src, T *dst, const int& src_stride, const int& dst_stride, const Args&... otherdims)
            {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
                // Memcpy specialization
                if (sizeof...(OTHER_COPYDIMS) == 0 && src_stride == 1 && dst_stride == 1) {
                    memcpy(dst, src, COPYDIM * sizeof(T) * VECLEN);
                    return;
                }
#endif

                __DACE_UNROLL
                for (int i = 0; i < COPYDIM; ++i)
                    CopyND<T, VECLEN, ALIGNED, OTHER_COPYDIMS...>::Dynamic::Copy(
                        src + i * src_stride, dst + i * dst_stride, otherdims...);
            }

            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc, const int& src_stride, const int& dst_stride, const Args&... otherdims)
            {
                __DACE_UNROLL
                for (int i = 0; i < COPYDIM; ++i)
                    CopyND<T, VECLEN, ALIGNED, OTHER_COPYDIMS...>::Dynamic::Accumulate(
                        src + i * src_stride, dst + i * dst_stride, acc, otherdims...);
            }
        };
    };
    
    // Specialization for actual copy / accumulation
    template <typename T, int VECLEN, int ALIGNED>
    struct CopyND<T, VECLEN, ALIGNED>
    {
        template <int...>
        struct ConstSrc
        {
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                *(vec<T, VECLEN> *)dst = *(vec<T, VECLEN> *)src;
            }

            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                *(vec<T, VECLEN> *)dst = acc(*(vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }
        };

        template <int...>
        struct ConstDst
        {
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                *(vec<T, VECLEN> *)dst = *(vec<T, VECLEN> *)src;
            }

            template <typename ACCUMULATE>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                *(vec<T, VECLEN> *)dst = acc(*(vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }
        };

        struct Dynamic
        {
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                *(vec<T, VECLEN> *)dst = *(vec<T, VECLEN> *)src;
            }

            template <typename ACCUMULATE>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                *(vec<T, VECLEN> *)dst = acc(*(vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }
        };
    };

    template <typename T, int VECLEN, int ALIGNED, int N>
    struct CopyNDDynamic
    {
        template <int SRC_STRIDE, int... OTHER_SRCDIMS>
        struct ConstSrc
        {
            template <typename... Args>
            static DACE_HDFI void Copy(const T *src, T *dst, const int& copydim, const int& dst_stride, const Args&... otherdims)
            {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
                // Memcpy specialization
                if (N == 1 && SRC_STRIDE == 1 && dst_stride == 1) {
                    memcpy(dst, src, copydim * sizeof(T) * VECLEN);
                    return;
                }
#endif

                __DACE_UNROLL
                for (int i = 0; i < copydim; ++i)
                    CopyNDDynamic<T, VECLEN, ALIGNED, N-1>::template ConstSrc<OTHER_SRCDIMS...>::Copy(
                        src + i * SRC_STRIDE, dst + i * dst_stride, otherdims...);
            }

            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc, const int& copydim, const int& dst_stride, const Args&... otherdims)
            {
                __DACE_UNROLL
                for (int i = 0; i < copydim; ++i)
                    CopyNDDynamic<T, VECLEN, ALIGNED, N-1>::template ConstSrc<OTHER_SRCDIMS...>::Accumulate(
                        src + i * SRC_STRIDE, dst + i * dst_stride, acc, otherdims...);
            }
        };

        template <int DST_STRIDE, int... OTHER_DSTDIMS>
        struct ConstDst
        {
            template <typename... Args>
            static DACE_HDFI void Copy(const T *src, T *dst, const int& copydim, const int& src_stride, const Args&... otherdims)
            {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
                // Memcpy specialization
                if (N == 1 && src_stride == 1 && DST_STRIDE == 1) {
                    memcpy(dst, src, copydim * sizeof(T) * VECLEN);
                    return;
                }
#endif

                __DACE_UNROLL
                for (int i = 0; i < copydim; ++i)
                    CopyNDDynamic<T, VECLEN, ALIGNED, N-1>::template ConstDst<OTHER_DSTDIMS...>::Copy(
                        src + i * src_stride, dst + i * DST_STRIDE, otherdims...);
            }

            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc, const int& copydim, const int& src_stride, const Args&... otherdims)
            {
                __DACE_UNROLL
                for (int i = 0; i < copydim; ++i)
                    CopyNDDynamic<T, VECLEN, ALIGNED, N-1>::template ConstDst<OTHER_DSTDIMS...>::Accumulate(
                        src + i * src_stride, dst + i * DST_STRIDE, acc, otherdims...);
            }
        };

        struct Dynamic
        {
            template <typename... Args>
            static DACE_HDFI void Copy(const T *src, T *dst, const int& copydim, const int& src_stride, const int& dst_stride, const Args&... otherdims)
            {
                static_assert(sizeof...(otherdims) == (N - 1) * 3, "Dimensionality mismatch in dynamic copy");

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
                // Memcpy specialization
                if (N == 1 && src_stride == 1 && dst_stride == 1) {
                    memcpy(dst, src, copydim * sizeof(T) * VECLEN);
                    return;
                }
#endif

                __DACE_UNROLL
                for (int i = 0; i < copydim; ++i)
                    CopyNDDynamic<T, VECLEN, ALIGNED, N - 1>::Dynamic::Copy(
                        src + i * src_stride, dst + i * dst_stride, otherdims...);
            }

            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc, const int& copydim, const int& src_stride, const int& dst_stride, const Args&... otherdims)
            {
                static_assert(sizeof...(otherdims) == (N - 1) * 3, "Dimensionality mismatch in dynamic copy");
                __DACE_UNROLL
                for (int i = 0; i < copydim; ++i)
                    CopyNDDynamic<T, VECLEN, ALIGNED, N - 1>::Dynamic::Accumulate(
                        src + i * src_stride, dst + i * dst_stride, acc, otherdims...);
            }
        };
    };

    template <typename T, int VECLEN, int ALIGNED>
    struct CopyNDDynamic<T, VECLEN, ALIGNED, 0>
    {
        template <int...>
        struct ConstSrc
        {
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                *(vec<T, VECLEN> *)dst = *(vec<T, VECLEN> *)src;
            }

            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                *(vec<T, VECLEN> *)dst = acc(*(vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }
        };

        template <int...>
        struct ConstDst
        {
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                *(vec<T, VECLEN> *)dst = *(vec<T, VECLEN> *)src;
            }

            template <typename ACCUMULATE>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                *(vec<T, VECLEN> *)dst = acc(*(vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }
        };

        struct Dynamic
        {
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                *(vec<T, VECLEN> *)dst = *(vec<T, VECLEN> *)src;
            }

            template <typename ACCUMULATE>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                *(vec<T, VECLEN> *)dst = acc(*(vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }
        };
    };

}  // namespace dace

#endif  // __DACE_COPY_H
