// Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_COPY_H
#define __DACE_COPY_H

#include "types.h"
#include "reduction.h"
#include "vector.h"

namespace dace
{
    template <typename T, int VECLEN,
              typename std::enable_if<std::is_trivially_copyable<T>::value, bool>::type = true>
    void CopyImpl(const T* src, T* dst, int COPYDIM)
    {
        memcpy(dst, src, COPYDIM * sizeof(T) * VECLEN);
    }

    template <typename T, int VECLEN,
              typename std::enable_if<!std::is_trivially_copyable<T>::value, bool>::type = true>
    void CopyImpl(const T* src, T* dst, int COPYDIM)
    {
        __DACE_UNROLL
        for (int i = 0; i < COPYDIM * VECLEN; ++i) {
            dst[i] = src[i];
        }
    }

    template <typename T, int COPYDIM, int VECLEN,
              typename std::enable_if<std::is_trivially_copyable<T>::value, bool>::type = true>
    void CopyImpl(const T* src, T* dst)
    {
        memcpy(dst, src, COPYDIM * sizeof(T) * VECLEN);
    }

    template <typename T, int COPYDIM, int VECLEN,
              typename std::enable_if<!std::is_trivially_copyable<T>::value, bool>::type = true>
    void CopyImpl(const T* src, T* dst)
    {
        __DACE_UNROLL
        for (int i = 0; i < COPYDIM * VECLEN; ++i) {
            dst[i] = src[i];
        }
    }

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
                    CopyImpl<T, COPYDIM, VECLEN>(src, dst);
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

            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc, const int& dst_stride, const Args&... dst_otherdims)
            {
                __DACE_UNROLL
                for (int i = 0; i < COPYDIM; ++i)
                    CopyND<T, VECLEN, ALIGNED, OTHER_COPYDIMS...>::template ConstSrc<OTHER_SRCDIMS...>::Accumulate_atomic(
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
                    CopyImpl<T, COPYDIM, VECLEN>(src, dst);
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

            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc, const int& src_stride, const Args&... src_otherdims)
            {
                __DACE_UNROLL
                for (int i = 0; i < COPYDIM; ++i)
                    CopyND<T, VECLEN, ALIGNED, OTHER_COPYDIMS...>::template ConstDst<OTHER_DSTDIMS...>::Accumulate_atomic(
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
                    CopyImpl<T, COPYDIM, VECLEN>(src, dst);
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

            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc, const int& src_stride, const int& dst_stride, const Args&... otherdims)
            {
                __DACE_UNROLL
                for (int i = 0; i < COPYDIM; ++i)
                    CopyND<T, VECLEN, ALIGNED, OTHER_COPYDIMS...>::Dynamic::Accumulate_atomic(
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
            template<typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                *(vec<T, VECLEN> *)dst = *(vec<T, VECLEN> *)src;
            }

            template<typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                __DACE_UNROLL
                for (int i = 0; i < VECLEN; i++){
                    dst[i] = src[i];
                }
            }

            template <typename ACCUMULATE, typename... Args,
                      typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                *(vec<T, VECLEN> *)dst = acc(*(vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }

            template <typename ACCUMULATE, typename... Args,
                      typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                __DACE_UNROLL
                for (int i = 0; i < VECLEN; i ++){
                    dst[i] = acc(dst[i], src[i]);
                }
            }

            template <typename ACCUMULATE, typename... Args,
                      typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc)
            {
                wcr_custom<T>::reduce_atomic(acc, (vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }

            template <typename ACCUMULATE, typename... Args,
                      typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc)
            {
                for (int i = 0; i < VECLEN; i++){
                    wcr_custom<T>::reduce_atomic(acc, dst, src);
                }
            }
        };

        template <int...>
        struct ConstDst
        {
            template <typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                *(vec<T, VECLEN> *)dst = *(vec<T, VECLEN> *)src;
            }

            template <typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                __DACE_UNROLL
                for (int i = 0; i < VECLEN; ++i){
                    dst[i] = src[i];
                }
            }

            template <typename ACCUMULATE, typename T2 = T,
                      std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                *(vec<T, VECLEN> *)dst = acc(*(vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }

            template <typename ACCUMULATE, typename T2 = T,
                      std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                for (int i = 0; i < VECLEN; ++i)
                    dst[i] = acc(dst[i], src[i]);
            }

            template <typename ACCUMULATE, typename T2 = T,
                    std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc)
            {
                wcr_custom<T>::reduce_atomic(acc, (vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }

            template <typename ACCUMULATE, typename T2 = T,
                     std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc)
            {
                for (int i = 0; i < VECLEN; i++){
                    wcr_custom<T>::reduce_atomic(acc, dst, src);
                }
            }
        };

        struct Dynamic
        {
            template <typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                *(vec<T, VECLEN> *)dst = *(vec<T, VECLEN> *)src;
            }

            template <typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                for (int i = 0; i < VECLEN; ++i){
                    dst[i] = src[i];
                }
            }

            template <typename ACCUMULATE, typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                *(vec<T, VECLEN> *)dst = acc(*(vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }

            template <typename ACCUMULATE, typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                for (int i = 0; i < VECLEN; ++i)
                    dst[i] = acc(dst[i], src[i]);
            }

            template <typename ACCUMULATE, typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc)
            {
                wcr_custom<T>::reduce_atomic(acc, (vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }

            template <typename ACCUMULATE, typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc)
            {
                wcr_custom<T>::reduce_atomic(acc, (vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
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
                    CopyImpl<T, VECLEN>(src, dst, copydim);
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

            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc, const int& copydim, const int& dst_stride, const Args&... otherdims)
            {
                __DACE_UNROLL
                for (int i = 0; i < copydim; ++i)
                    CopyNDDynamic<T, VECLEN, ALIGNED, N-1>::template ConstSrc<OTHER_SRCDIMS...>::Accumulate_atomic(
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
                    CopyImpl<T, VECLEN>(src, dst, copydim);
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

            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc, const int& copydim, const int& src_stride, const Args&... otherdims)
            {
                __DACE_UNROLL
                for (int i = 0; i < copydim; ++i)
                    CopyNDDynamic<T, VECLEN, ALIGNED, N-1>::template ConstDst<OTHER_DSTDIMS...>::Accumulate_atomic(
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
                    CopyImpl<T, VECLEN>(src, dst, copydim);
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

            template <typename ACCUMULATE, typename... Args>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc, const int& copydim, const int& src_stride, const int& dst_stride, const Args&... otherdims)
            {
                static_assert(sizeof...(otherdims) == (N - 1) * 3, "Dimensionality mismatch in dynamic copy");
                __DACE_UNROLL
                for (int i = 0; i < copydim; ++i)
                    CopyNDDynamic<T, VECLEN, ALIGNED, N - 1>::Dynamic::Accumulate_atomic(
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
            template<typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                *(vec<T, VECLEN> *)dst = *(vec<T, VECLEN> *)src;

            }

            template <typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                for (int i = 0; i < VECLEN; ++i){
                    dst[i] = src[i];
                }
            }

            template <typename ACCUMULATE, typename... Args,
                      typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                *(vec<T, VECLEN> *)dst = acc(*(vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }

            template <typename ACCUMULATE, typename... Args, typename T2 = T,
                      std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc, Args... args)
            {
                for (int i = 0; i < VECLEN; ++i)
                    dst[i] = acc(dst[i], src[i]);
            }

            template <typename ACCUMULATE, typename... Arg,
                      typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc)
            {
                wcr_custom<T>::reduce_atomic(acc, (vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }

            template <typename ACCUMULATE, typename... Args,
                    typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc, Args... args)
            {
                for (int i = 0; i < VECLEN; ++i)
                    wcr_custom<T>::reduce_atomic(acc, &dst[i], src[i]);
            }

        };

        template <int...>
        struct ConstDst
        {
            template <typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                *(vec<T, VECLEN> *)dst = *(vec<T, VECLEN> *)src;

            }

            template <typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                for (int i = 0; i < VECLEN; ++i)
                    dst[i] = src[i];

            }

            template <typename ACCUMULATE,
                      typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                *(vec<T, VECLEN> *)dst = acc(*(vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }

            template <typename ACCUMULATE,
                      typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                for (int i = 0; i < VECLEN; i++){
                    dst[i] = acc(dst[i], src[i]);
                }
            }

            template <typename ACCUMULATE,
                      typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc)
            {
                wcr_custom<T>::reduce_atomic(acc, (vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }

            template <typename ACCUMULATE,
                      typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc)
            {
                for (int i = 0; i < VECLEN; i++){
                    wcr_custom<T>::reduce_atomic(acc, dst[i], src[i]);
                }
            }
        };

        struct Dynamic
        {
            template <typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                *(vec<T, VECLEN> *)dst = *(vec<T, VECLEN> *)src;

            }

            template <typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Copy(const T *src, T *dst)
            {
                for (int i = 0; i < VECLEN; i++){
                    dst[i] = src[i];
                }
            }

            template <typename ACCUMULATE,
                      typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                *(vec<T, VECLEN> *)dst = acc(*(vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }

            template <typename ACCUMULATE,
                      typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate(const T *src, T *dst, ACCUMULATE acc)
            {
                for (int i = 0; i < VECLEN; i++){
                    dst = acc(dst[i], src[i]);
                }
            }

            template <typename ACCUMULATE,
                      typename T2 = T, std::enable_if_t<std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc)
            {
                wcr_custom<T>::reduce_atomic(acc, (vec<T, VECLEN> *)dst, *(vec<T, VECLEN> *)src);
            }

            template <typename ACCUMULATE,
                      typename T2 = T, std::enable_if_t<!std::is_trivially_copyable<T2>::value, bool> = true>
            static DACE_HDFI void Accumulate_atomic(const T *src, T *dst, ACCUMULATE acc)
            {
                for (int i = 0; i < VECLEN; i++){
                    wcr_custom<T>::reduce_atomic(acc, dst[i], src[i]);
                }
            }
        };
    };

}  // namespace dace

#endif  // __DACE_COPY_H
