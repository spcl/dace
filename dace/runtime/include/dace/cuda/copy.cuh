// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

//------------------------------------------------------------------------
// Adapted from "MAPS: GPU Optimization and Memory Abstraction Framework"
// https://github.com/maps-gpu/MAPS
// Copyright (c) 2015, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------

#ifndef __DACE_CUDACOPY_CUH
#define __DACE_CUDACOPY_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include "../types.h"
#include "../vector.h"
#include "../reduction.h"


namespace dace
{

    // Converts from an integral amount of bytes to a type.
    template <int BYTES>
    struct BytesToType
    {
        typedef void type;
    };

    #ifdef __DACE_BYTES_TO_TYPE
    #error Using disallowed macro name __DACE_BYTES_TO_TYPE
    #endif

    #define __DACE_BYTES_TO_TYPE(bytes, t)                  \
    template<>                                              \
    struct BytesToType<bytes>                               \
    {                                                       \
        typedef t type;                                     \
    }

    __DACE_BYTES_TO_TYPE(16, float4);
    __DACE_BYTES_TO_TYPE(8, uint64_t);
    __DACE_BYTES_TO_TYPE(4, uint32_t);
    __DACE_BYTES_TO_TYPE(2, uint16_t);
    __DACE_BYTES_TO_TYPE(1, uint8_t);

    # undef __DACE_BYTES_TO_TYPE

    template<unsigned int BLOCK_WIDTH, unsigned int BLOCK_HEIGHT, unsigned int BLOCK_DEPTH>
    struct LinearizeTID
    {
        static DACE_DFI unsigned int get()
        {
            return threadIdx.x + threadIdx.y * BLOCK_WIDTH + 
                threadIdx.z * BLOCK_WIDTH * BLOCK_HEIGHT;
        }
    };

    template<unsigned int BLOCK_WIDTH, unsigned int BLOCK_HEIGHT>
    struct LinearizeTID<BLOCK_WIDTH, BLOCK_HEIGHT, 1>
    {
        static DACE_DFI unsigned int get()
        {
            return threadIdx.x + threadIdx.y * BLOCK_WIDTH;
        }
    };

    template<unsigned int BLOCK_WIDTH>
    struct LinearizeTID<BLOCK_WIDTH, 1, 1>
    {
        static DACE_DFI unsigned int get()
        {
            return threadIdx.x;
        }
    };

    template<int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH>
    static DACE_DFI unsigned int GetLinearTID() {
        return LinearizeTID<BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH>::get();
    }

    ////////////////////////////////////////////////////////////////////////
    // Detect optimal bit read preference

    enum
    {
        #if __CUDA_ARCH__ >= 500
        PREFERRED_GREAD_SIZE = 128 / 8, // 128-bit
        PREFERRED_SWRITE_SIZE = 128 / 8, // 128-bit
        #elif __CUDA_ARCH__ >= 300
        PREFERRED_GREAD_SIZE = 128 / 8, // 128-bit
        PREFERRED_SWRITE_SIZE = 64 / 8, // 64-bit
        #elif __CUDA_ARCH__ >= 130
        PREFERRED_GREAD_SIZE = 64 / 8, // 64-bit
        PREFERRED_SWRITE_SIZE = 32 / 8, // 32-bit
        #else
        PREFERRED_GREAD_SIZE = 32 / 8, // Default to 32-bit loads
        PREFERRED_SWRITE_SIZE = 32 / 8, // 32-bit
        #endif
    };

    #define DEBUG_PRINT(...) do {} while(0)
    #define BLOCK_PRINT(...) do {} while(0)

    //#define DEBUG_PRINT(...) do { if(threadIdx.x + threadIdx.y == 0 && blockIdx.x + blockIdx.y + blockIdx.z == 0 && threadIdx.z == 1) printf(__VA_ARGS__);  } while(0)
    //#define BLOCK_PRINT(...) do { if(blockIdx.x + blockIdx.y + blockIdx.z == 0) printf(__VA_ARGS__);  } while(0)

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
              int COPY_ZLEN, int COPY_YLEN, int COPY_XLEN, 
              int DST_ZSTRIDE, int DST_YSTRIDE, int DST_XSTRIDE,
              bool ASYNC>
    static DACE_DFI void GlobalToShared3D(
            const T *ptr, int src_zstride,
            int src_ystride, int src_xstride, T *smem)
    {       
        // Linear thread ID
        int ltid = GetLinearTID<BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH>();
        
        constexpr int BLOCK_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * BLOCK_DEPTH;
        constexpr int TOTAL_XYZ = COPY_XLEN * COPY_YLEN * COPY_ZLEN;
        constexpr int TOTAL_XY = COPY_XLEN * COPY_YLEN;
        constexpr int XY_SLICES = BLOCK_SIZE / TOTAL_XY;
        constexpr int XY_REM = BLOCK_SIZE % TOTAL_XY;
        constexpr int X_SLICES = BLOCK_SIZE / COPY_XLEN;
        constexpr int X_REM = BLOCK_SIZE % COPY_XLEN;

        //////////////////////////////////////////////////////////////////////
        // Block size larger than number of elements, one read
        if ((BLOCK_SIZE / TOTAL_XYZ) > 0)
        {
            DEBUG_PRINT("Chose path XYZ\n");

            // De-linearize
            int ltidx = ltid % COPY_XLEN;
            int ltidy = (ltid / COPY_XLEN) % COPY_YLEN;
            int ltidz = (ltid / COPY_XLEN) / COPY_YLEN;

            if (ltid < TOTAL_XYZ)
            {
                smem[ltidx*DST_XSTRIDE + ltidy * DST_YSTRIDE + ltidz * DST_ZSTRIDE] =
                    *(ptr + ltidx * src_xstride
                      + src_ystride * ltidy
                      + src_zstride * ltidz);
            }
        }

        //////////////////////////////////////////////////////////////////////
        // More than one XY slice
        else if ((BLOCK_SIZE / TOTAL_XYZ) == 0 && XY_SLICES > 0 && XY_REM > 0)
        {
            DEBUG_PRINT("Chose path XY.1\n");

            // Currently, only use threads in slice
            // TODO(later): If contiguous (DST_YSTRIDE == COPY_XLEN), use the rest
            constexpr int SLICES_PER_ITER = (XY_SLICES == 0 ? 1 :  // Compilers are annoying
                                             COPY_ZLEN / XY_SLICES);
            constexpr int REMAINDER = (XY_SLICES == 0 ? 1 : // Compilers are annoying
                                       COPY_ZLEN % XY_SLICES);
            constexpr int REMOFF = SLICES_PER_ITER * XY_SLICES;

            if (ltid < (BLOCK_SIZE - XY_REM))
            {
                // De-linearize
                int ltidx = ltid % COPY_XLEN;
                int ltidy = (ltid / COPY_XLEN) % COPY_YLEN;
                int ltidz = (ltid / COPY_XLEN) / COPY_YLEN;

                #pragma unroll
                for (int i = 0; i < SLICES_PER_ITER; ++i)
                {
                    smem[ltidx * DST_XSTRIDE + ltidy * DST_YSTRIDE + (i*XY_SLICES + ltidz) * DST_ZSTRIDE] =
                        *(ptr +
                          src_xstride * ltidx +
                          src_ystride * ltidy +
                          src_zstride * (i * XY_SLICES + ltidz));
                }

                if (ltidz < REMAINDER)
                {
                    // Read remainder
                    smem[ltidx * DST_XSTRIDE + ltidy * DST_YSTRIDE + (REMOFF + ltidz) * DST_ZSTRIDE] =
                        *(ptr +
                          src_xstride * ltidx +
                          src_ystride * ltidy +
                          src_zstride * (REMOFF + ltidz));
                }
            }
        }

        //////////////////////////////////////////////////////////////////////
        // Exactly n*XY slices
        else if ((BLOCK_SIZE / TOTAL_XYZ) == 0 && XY_SLICES > 0 && XY_REM == 0)
        {
            DEBUG_PRINT("Chose path XY.2\n");

            constexpr int SLICES_PER_ITER = (XY_SLICES == 0 ? 1 :  // Compilers are annoying
                                             COPY_ZLEN / XY_SLICES);
            constexpr int REMAINDER = (XY_SLICES == 0 ? 1 :  // Compilers are annoying
                                       COPY_ZLEN % XY_SLICES);
            constexpr int REMOFF = SLICES_PER_ITER * XY_SLICES;

            // De-linearize
            int ltidx = ltid % COPY_XLEN;
            int ltidy = (ltid / COPY_XLEN) % COPY_YLEN;
            int ltidz = (ltid / COPY_XLEN) / COPY_YLEN;

            #pragma unroll
            for (int i = 0; i < SLICES_PER_ITER; ++i)
            {
                smem[ltidx * DST_XSTRIDE + ltidy * DST_YSTRIDE + (i*XY_SLICES + ltidz) * DST_ZSTRIDE] =
                    *(ptr +
                      src_xstride * ltidx +
                      src_ystride * ltidy +
                      src_zstride * (i * XY_SLICES + ltidz));
            }

            if (ltidz < REMAINDER)
            {
                // Read remainder
                smem[ltidx * DST_XSTRIDE + ltidy * DST_YSTRIDE + (REMOFF + ltidz) * DST_ZSTRIDE] =
                    *(ptr +
                      src_xstride * ltidx +
                      src_ystride * ltidy +
                      src_zstride * (REMOFF + ltidz));
            }
        }

        //////////////////////////////////////////////////////////////////////
        // More than X row
        else if (XY_SLICES == 0 && X_SLICES > 0 && X_REM > 0)
        {
            DEBUG_PRINT("Chose path X.1\n");

            // Currently, only use threads in row
            // TODO(later): If contiguous (DST_YSTRIDE == COPY_XLEN), use the rest
            constexpr int ROWS_PER_XY_SLICE = (X_SLICES == 0 ? 1 :  // Compilers are annoying
                                               COPY_YLEN / X_SLICES);
            constexpr int REMAINDER = (X_SLICES == 0 ? 1 :  // Compilers are annoying
                                       COPY_YLEN % X_SLICES);
            constexpr int REMOFF = ROWS_PER_XY_SLICE * X_SLICES;

            if (ltid < (BLOCK_SIZE - X_REM))
            {
                // De-linearize
                int ltidx = ltid % COPY_XLEN;
                int ltidy = ltid / COPY_XLEN;

                #pragma unroll
                for (int i = 0; i < COPY_ZLEN; ++i)
                {
                    #pragma unroll
                    for (int j = 0; j < ROWS_PER_XY_SLICE; ++j)
                    {
                        smem[ltidx * DST_XSTRIDE + (j*X_SLICES + ltidy) * DST_YSTRIDE + i * DST_ZSTRIDE] =
                            *(ptr +
                              src_xstride * ltidx +
                              src_ystride * (j * X_SLICES + ltidy) +
                              src_zstride * i);
                    }

                    if (ltidy < REMAINDER)
                    {
                        // Read remainder
                        smem[ltidx * DST_XSTRIDE + (REMOFF + ltidy)* DST_YSTRIDE + i * DST_ZSTRIDE] =
                            *(ptr +
                              src_xstride * ltidx +
                              src_ystride * (REMOFF + ltidy) +
                              src_zstride * i);
                    }

                }
            }
        }

        //////////////////////////////////////////////////////////////////////
        // Exactly n*X rows
        else if (XY_SLICES == 0 && X_SLICES > 0 && X_REM == 0)
        {
            DEBUG_PRINT("Chose path X.2\n");

            constexpr int ROWS_PER_XY_SLICE = (X_SLICES == 0 ? 1 :  // Compilers are annoying
                                               COPY_YLEN / X_SLICES);
            constexpr int REMAINDER = (X_SLICES == 0 ? 1 :  // Compilers are annoying
                                       COPY_YLEN % X_SLICES);
            constexpr int REMOFF = ROWS_PER_XY_SLICE * X_SLICES;

            // De-linearize
            int ltidx = ltid % COPY_XLEN;
            int ltidy = ltid / COPY_XLEN;

            #pragma unroll
            for (int i = 0; i < COPY_ZLEN; ++i)
            {
                #pragma unroll
                for (int j = 0; j < ROWS_PER_XY_SLICE; ++j)
                {
                    smem[ltidx * DST_XSTRIDE + (j*X_SLICES + ltidy) * DST_YSTRIDE + i * DST_ZSTRIDE] =
                        *(ptr +
                          src_xstride * ltidx +
                          src_ystride * (j * X_SLICES + ltidy) +
                          src_zstride * i);
                }

                if (ltidy < REMAINDER)
                {
                    // Read remainder
                    smem[ltidx * DST_XSTRIDE + (REMOFF + ltidy) * DST_YSTRIDE + i * DST_ZSTRIDE] =
                        *(ptr +
                          src_xstride * ltidx +
                          src_ystride * (REMOFF + ltidy) +
                          src_zstride * i);
                }

            }
        }

        //////////////////////////////////////////////////////////////////////
        // Less than one X row
        else if (X_SLICES == 0)
        {
            DEBUG_PRINT("Chose path X.3\n");


            constexpr int ITERATIONS_PER_ROW = COPY_XLEN / BLOCK_SIZE;
            constexpr int REMAINDER = COPY_XLEN % BLOCK_SIZE;
            constexpr int REMOFF = ITERATIONS_PER_ROW * BLOCK_SIZE;

            #pragma unroll
            for (int i = 0; i < COPY_ZLEN; ++i)
            {
                #pragma unroll
                for (int j = 0; j < COPY_YLEN; ++j)
                {
                    #pragma unroll
                    for (int k = 0; k < ITERATIONS_PER_ROW; ++k)
                    {
                        smem[(k * BLOCK_SIZE + ltid) * DST_XSTRIDE + j * DST_YSTRIDE + i * DST_ZSTRIDE] =
                            *(ptr +
                              src_xstride * (k * BLOCK_SIZE + ltid) +
                              src_ystride * j +
                              src_zstride * i);
                    }

                    if (ltid < REMAINDER)
                    {
                        // Read remainder
                        smem[(REMOFF + ltid) * DST_ZSTRIDE + j * DST_YSTRIDE + i * DST_ZSTRIDE] =
                            *(ptr +
                              src_xstride * (REMOFF + ltid) +
                              src_ystride * j +
                              src_zstride * i);
                    }
                }
            }
        }

        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////


        if (!ASYNC)
            __syncthreads();
    }

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
              int COPY_XLEN, int DST_XSTRIDE,
              bool ASYNC>
    static DACE_DFI void GlobalToShared1D(
            const T *ptr, int src_xstride, T *smem)
    {
        GlobalToShared3D<T, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 1,
            1, COPY_XLEN, 1, 1, DST_XSTRIDE, ASYNC>(
                ptr, 1, 1, src_xstride, smem);
    }
    
    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
              int COPY_YLEN, int COPY_XLEN, int DST_YSTRIDE, int DST_XSTRIDE,
              bool ASYNC>
    static DACE_DFI void GlobalToShared2D(
            const T *ptr, int src_ystride, int src_xstride,
            T *smem)
    {
        GlobalToShared3D<T, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 1, 
                         COPY_YLEN, COPY_XLEN, 1, DST_YSTRIDE, DST_XSTRIDE, 
                         ASYNC>(
            ptr, 1, src_ystride, src_xstride, smem);
    }

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
              bool ASYNC>
        static DACE_DFI void GlobalToShared3DDynamic(
            const T *ptr, int src_zstride,
            int src_ystride, int src_xstride, T *smem, int DST_ZSTRIDE, int DST_YSTRIDE, int DST_XSTRIDE,
            int COPY_ZLEN, int COPY_YLEN, int COPY_XLEN)
    {
        // Linear thread ID
        int ltid = GetLinearTID<BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH>();

        int BLOCK_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * BLOCK_DEPTH;
        int TOTAL_XYZ = COPY_XLEN * COPY_YLEN * COPY_ZLEN;
        int TOTAL_XY = COPY_XLEN * COPY_YLEN;
        int XY_SLICES = BLOCK_SIZE / TOTAL_XY;
        int XY_REM = BLOCK_SIZE % TOTAL_XY;
        int X_SLICES = BLOCK_SIZE / COPY_XLEN;
        int X_REM = BLOCK_SIZE % COPY_XLEN;

        //////////////////////////////////////////////////////////////////////
        // Block size larger than number of elements, one read
        if ((BLOCK_SIZE / TOTAL_XYZ) > 0)
        {
            DEBUG_PRINT("Chose path XYZ\n");

            // De-linearize
            int ltidx = ltid % COPY_XLEN;
            int ltidy = (ltid / COPY_XLEN) % COPY_YLEN;
            int ltidz = (ltid / COPY_XLEN) / COPY_YLEN;

            if (ltid < TOTAL_XYZ)
            {
                smem[ltidx*DST_XSTRIDE + ltidy * DST_YSTRIDE + ltidz * DST_ZSTRIDE] =
                    *(ptr + ltidx * src_xstride
                      + src_ystride * ltidy
                      + src_zstride * ltidz);
            }
        }

        //////////////////////////////////////////////////////////////////////
        // More than one XY slice
        else if ((BLOCK_SIZE / TOTAL_XYZ) == 0 && XY_SLICES > 0 && XY_REM > 0)
        {
            DEBUG_PRINT("Chose path XY.1\n");

            // Currently, only use threads in slice
            // TODO(later): If contiguous (DST_YSTRIDE == COPY_XLEN), use the rest
            int SLICES_PER_ITER = (XY_SLICES == 0 ? 1 :  // Compilers are annoying
                                             COPY_ZLEN / XY_SLICES);
            int REMAINDER = (XY_SLICES == 0 ? 1 : // Compilers are annoying
                                       COPY_ZLEN % XY_SLICES);
            int REMOFF = SLICES_PER_ITER * XY_SLICES;

            if (ltid < (BLOCK_SIZE - XY_REM))
            {
                // De-linearize
                int ltidx = ltid % COPY_XLEN;
                int ltidy = (ltid / COPY_XLEN) % COPY_YLEN;
                int ltidz = (ltid / COPY_XLEN) / COPY_YLEN;

                #pragma unroll
                for (int i = 0; i < SLICES_PER_ITER; ++i)
                {
                    smem[ltidx * DST_XSTRIDE + ltidy * DST_YSTRIDE + (i*XY_SLICES + ltidz) * DST_ZSTRIDE] =
                        *(ptr +
                          src_xstride * ltidx +
                          src_ystride * ltidy +
                          src_zstride * (i * XY_SLICES + ltidz));
                }

                if (ltidz < REMAINDER)
                {
                    // Read remainder
                    smem[ltidx * DST_XSTRIDE + ltidy * DST_YSTRIDE + (REMOFF + ltidz) * DST_ZSTRIDE] =
                        *(ptr +
                          src_xstride * ltidx +
                          src_ystride * ltidy +
                          src_zstride * (REMOFF + ltidz));
                }
            }
        }

        //////////////////////////////////////////////////////////////////////
        // Exactly n*XY slices
        else if ((BLOCK_SIZE / TOTAL_XYZ) == 0 && XY_SLICES > 0 && XY_REM == 0)
        {
            DEBUG_PRINT("Chose path XY.2\n");

            int SLICES_PER_ITER = (XY_SLICES == 0 ? 1 :  // Compilers are annoying
                                             COPY_ZLEN / XY_SLICES);
            int REMAINDER = (XY_SLICES == 0 ? 1 :  // Compilers are annoying
                                       COPY_ZLEN % XY_SLICES);
            int REMOFF = SLICES_PER_ITER * XY_SLICES;

            // De-linearize
            int ltidx = ltid % COPY_XLEN;
            int ltidy = (ltid / COPY_XLEN) % COPY_YLEN;
            int ltidz = (ltid / COPY_XLEN) / COPY_YLEN;

            #pragma unroll
            for (int i = 0; i < SLICES_PER_ITER; ++i)
            {
                smem[ltidx * DST_XSTRIDE + ltidy * DST_YSTRIDE + (i*XY_SLICES + ltidz) * DST_ZSTRIDE] =
                    *(ptr +
                      src_xstride * ltidx +
                      src_ystride * ltidy +
                      src_zstride * (i * XY_SLICES + ltidz));
            }

            if (ltidz < REMAINDER)
            {
                // Read remainder
                smem[ltidx * DST_XSTRIDE + ltidy * DST_YSTRIDE + (REMOFF + ltidz) * DST_ZSTRIDE] =
                    *(ptr +
                      src_xstride * ltidx +
                      src_ystride * ltidy +
                      src_zstride * (REMOFF + ltidz));
            }
        }

        //////////////////////////////////////////////////////////////////////
        // More than X row
        else if (XY_SLICES == 0 && X_SLICES > 0 && X_REM > 0)
        {
            DEBUG_PRINT("Chose path X.1\n");

            // Currently, only use threads in row
            // TODO(later): If contiguous (DST_YSTRIDE == COPY_XLEN), use the rest
            int ROWS_PER_XY_SLICE = (X_SLICES == 0 ? 1 :  // Compilers are annoying
                                               COPY_YLEN / X_SLICES);
            int REMAINDER = (X_SLICES == 0 ? 1 :  // Compilers are annoying
                                       COPY_YLEN % X_SLICES);
            int REMOFF = ROWS_PER_XY_SLICE * X_SLICES;

            if (ltid < (BLOCK_SIZE - X_REM))
            {
                // De-linearize
                int ltidx = ltid % COPY_XLEN;
                int ltidy = ltid / COPY_XLEN;

                #pragma unroll
                for (int i = 0; i < COPY_ZLEN; ++i)
                {
                    #pragma unroll
                    for (int j = 0; j < ROWS_PER_XY_SLICE; ++j)
                    {
                        smem[ltidx * DST_XSTRIDE + (j*X_SLICES + ltidy) * DST_YSTRIDE + i * DST_ZSTRIDE] =
                            *(ptr +
                              src_xstride * ltidx +
                              src_ystride * (j * X_SLICES + ltidy) +
                              src_zstride * i);
                    }

                    if (ltidy < REMAINDER)
                    {
                        // Read remainder
                        smem[ltidx * DST_XSTRIDE + (REMOFF + ltidy)* DST_YSTRIDE + i * DST_ZSTRIDE] =
                            *(ptr +
                              src_xstride * ltidx +
                              src_ystride * (REMOFF + ltidy) +
                              src_zstride * i);
                    }

                }
            }
        }

        //////////////////////////////////////////////////////////////////////
        // Exactly n*X rows
        else if (XY_SLICES == 0 && X_SLICES > 0 && X_REM == 0)
        {
            DEBUG_PRINT("Chose path X.2\n");

            int ROWS_PER_XY_SLICE = (X_SLICES == 0 ? 1 :  // Compilers are annoying
                                               COPY_YLEN / X_SLICES);
            int REMAINDER = (X_SLICES == 0 ? 1 :  // Compilers are annoying
                                       COPY_YLEN % X_SLICES);
            int REMOFF = ROWS_PER_XY_SLICE * X_SLICES;

            // De-linearize
            int ltidx = ltid % COPY_XLEN;
            int ltidy = ltid / COPY_XLEN;

            #pragma unroll
            for (int i = 0; i < COPY_ZLEN; ++i)
            {
                #pragma unroll
                for (int j = 0; j < ROWS_PER_XY_SLICE; ++j)
                {
                    smem[ltidx * DST_XSTRIDE + (j*X_SLICES + ltidy) * DST_YSTRIDE + i * DST_ZSTRIDE] =
                        *(ptr +
                          src_xstride * ltidx +
                          src_ystride * (j * X_SLICES + ltidy) +
                          src_zstride * i);
                }

                if (ltidy < REMAINDER)
                {
                    // Read remainder
                    smem[ltidx * DST_XSTRIDE + (REMOFF + ltidy) * DST_YSTRIDE + i * DST_ZSTRIDE] =
                        *(ptr +
                          src_xstride * ltidx +
                          src_ystride * (REMOFF + ltidy) +
                          src_zstride * i);
                }

            }
        }

        //////////////////////////////////////////////////////////////////////
        // Less than one X row
        else if (X_SLICES == 0)
        {
            DEBUG_PRINT("Chose path X.3\n");


            int ITERATIONS_PER_ROW = COPY_XLEN / BLOCK_SIZE;
            int REMAINDER = COPY_XLEN % BLOCK_SIZE;
            int REMOFF = ITERATIONS_PER_ROW * BLOCK_SIZE;

            #pragma unroll
            for (int i = 0; i < COPY_ZLEN; ++i)
            {
                #pragma unroll
                for (int j = 0; j < COPY_YLEN; ++j)
                {
                    #pragma unroll
                    for (int k = 0; k < ITERATIONS_PER_ROW; ++k)
                    {
                        smem[(k * BLOCK_SIZE + ltid) * DST_XSTRIDE + j * DST_YSTRIDE + i * DST_ZSTRIDE] =
                            *(ptr +
                              src_xstride * (k * BLOCK_SIZE + ltid) +
                              src_ystride * j +
                              src_zstride * i);
                    }

                    if (ltid < REMAINDER)
                    {
                        // Read remainder
                        smem[(REMOFF + ltid) * DST_ZSTRIDE + j * DST_YSTRIDE + i * DST_ZSTRIDE] =
                            *(ptr +
                              src_xstride * (REMOFF + ltid) +
                              src_ystride * j +
                              src_zstride * i);
                    }
                }
            }
        }

        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////


        if (!ASYNC)
            __syncthreads();
    }

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
        bool ASYNC>
        static DACE_DFI void GlobalToShared1DDynamic(
            const T *ptr, int src_xstride, T *smem, int DST_XSTRIDE, int COPY_XLEN)
    {
        GlobalToShared3DDynamic<T, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, ASYNC>(
                ptr, 1, 1, src_xstride, smem, 1, 1, DST_XSTRIDE, 1, 1, COPY_XLEN);
    }

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
        bool ASYNC>
        static DACE_DFI void GlobalToShared2DDynamic(
            const T *ptr, int src_ystride, int src_xstride,
            T *smem, int DST_YSTRIDE, int DST_XSTRIDE, int COPY_YLEN, int COPY_XLEN)
    {
        GlobalToShared3DDynamic<T, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, ASYNC>(
                ptr, 1, src_ystride, src_xstride, smem, 1, DST_YSTRIDE, DST_XSTRIDE, 1, COPY_YLEN, COPY_XLEN);
    }


    /*
    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
        int COPY_XLEN, int DST_XSTRIDE,
        bool ASYNC>
        static DACE_DFI void SharedToGlobal1D(
            const T *smem, int src_xstride, T *ptr)
    {
        GlobalToShared3D<T, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 1,
            1, COPY_XLEN, 1, 1, DST_XSTRIDE, ASYNC>(
                smem, 1, 1, src_xstride, ptr);
    }
    */

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
        int SMEM_TOTAL_ELEMENTS, int DST_XSTRIDE,
        bool ASYNC>
    struct ResetShared
    {
        static DACE_DFI void Reset(T *smem_) {
            int *smem = (int *)smem_;
            // Linear thread ID
            int ltid = GetLinearTID<BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH>();
            constexpr int BLOCK_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * BLOCK_DEPTH;
            constexpr int TOTAL = (SMEM_TOTAL_ELEMENTS * sizeof(T)) / sizeof(int);
            constexpr int WRITES = TOTAL / BLOCK_SIZE;
            constexpr int REM_WRITES = TOTAL % BLOCK_SIZE;

            #pragma unroll
            for (int i = 0; i < WRITES; ++i) {
                *(smem + (ltid + i * BLOCK_SIZE) * DST_XSTRIDE) = 0;
            }

            if (REM_WRITES != 0) {
                if (ltid < REM_WRITES)
                    *(smem + (ltid + WRITES * BLOCK_SIZE) * DST_XSTRIDE) = 0;
            }

            if (!ASYNC)
                __syncthreads();
        }
    };

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
        int COPY_XLEN, bool ASYNC>
    struct SharedToGlobal1D
    {
        template <typename WCR>
        static DACE_DFI void Accum(const T *smem, int src_xstride, T *ptr, int DST_XSTRIDE, WCR wcr)
        {
            if (!ASYNC)
                __syncthreads();

            // Linear thread ID
            int ltid = GetLinearTID<BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH>();
            constexpr int BLOCK_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * BLOCK_DEPTH;
            constexpr int TOTAL = COPY_XLEN;
            constexpr int WRITES = TOTAL / BLOCK_SIZE;
            constexpr int REM_WRITES = TOTAL % BLOCK_SIZE;

            #pragma unroll
            for (int i = 0; i < WRITES; ++i) {
                wcr_custom<T>::template reduce(
                    wcr, ptr + (ltid + i * BLOCK_SIZE) * DST_XSTRIDE,
                    *(smem + (ltid + i * BLOCK_SIZE) * src_xstride));
            }

            if (REM_WRITES != 0) {
                if (ltid < REM_WRITES)
                    wcr_custom<T>::template reduce(
                        ptr + (ltid + WRITES * BLOCK_SIZE)* DST_XSTRIDE,
                        *(smem + (ltid + WRITES * BLOCK_SIZE) * src_xstride));
            }
        }

        template <ReductionType REDTYPE>
        static DACE_DFI void Accum(const T *smem, int src_xstride, T *ptr, int DST_XSTRIDE)
        {
            if (!ASYNC)
                __syncthreads();
            
            // Linear thread ID
            int ltid = GetLinearTID<BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH>();
            constexpr int BLOCK_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * BLOCK_DEPTH;
            constexpr int TOTAL = COPY_XLEN;
            constexpr int WRITES = TOTAL / BLOCK_SIZE;
            constexpr int REM_WRITES = TOTAL % BLOCK_SIZE;

            #pragma unroll
            for (int i = 0; i < WRITES; ++i) {
                wcr_fixed<REDTYPE, T>::template reduce_atomic(
                    ptr + (ltid + i * BLOCK_SIZE) * DST_XSTRIDE,
                    *(smem + (ltid + i * BLOCK_SIZE) * src_xstride));
            }

            if (REM_WRITES != 0) {
                if (ltid < REM_WRITES)
                    wcr_fixed<REDTYPE, T>::template reduce_atomic(
                        ptr + (ltid + WRITES*BLOCK_SIZE)* DST_XSTRIDE,
                        *(smem + (ltid + WRITES * BLOCK_SIZE) * src_xstride));
            }
        }
    };
    
    // TODO: Make like SharedToGlobal1D
    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
        int COPY_YLEN, int COPY_XLEN, int DST_YSTRIDE, int DST_XSTRIDE,
        bool ASYNC>
    static DACE_DFI void SharedToGlobal2D(
            const T *smem, int src_ystride, int src_xstride,
            T *ptr)
    {
        GlobalToShared3D<T, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 1,
            COPY_YLEN, COPY_XLEN, 1, DST_YSTRIDE, DST_XSTRIDE,
            ASYNC>(
                smem, 1, src_ystride, src_xstride, ptr);
    }
    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, bool ASYNC>
    static DACE_DFI void SharedToGlobal2DDynamic(
            const T *smem, int src_ystride, int src_xstride,
            T *ptr, int DST_YSTRIDE, int DST_XSTRIDE, int COPY_YLEN, int COPY_XLEN)
    {
        GlobalToShared3DDynamic<T, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, ASYNC>(
                smem, 1, src_ystride, src_xstride, ptr, 1,
                DST_YSTRIDE, DST_XSTRIDE, 1, COPY_YLEN, COPY_XLEN);
    }

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH,
              int COPY_XLEN, int DST_XSTRIDE,
              bool ASYNC>
    static DACE_DFI void GlobalToGlobal1D(
            const T *src, int src_xstride, T *dst)
    {
        if (src_xstride == 1)
        {
	    __DACE_UNROLL
	    for (int i = 0; i < COPY_XLEN; ++i)
	        dst[i*DST_XSTRIDE] = src[i];
        }
        else
	{
	    __DACE_UNROLL
	    for (int i = 0; i < COPY_XLEN; ++i)
	        dst[i*DST_XSTRIDE] = src[i*src_xstride];
	}
    }

    template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, bool ASYNC>
    static DACE_DFI void GlobalToGlobal1DDynamic(
            const T *src, int src_xstride, T *dst, int DST_XSTRIDE, int COPY_XLEN)
    {
        if (src_xstride == 1)
        {
	    __DACE_UNROLL
	    for (int i = 0; i < COPY_XLEN; ++i)
	        dst[i*DST_XSTRIDE] = src[i];
        }
        else
	{
	    __DACE_UNROLL
	    for (int i = 0; i < COPY_XLEN; ++i)
	        dst[i*DST_XSTRIDE] = src[i*src_xstride];
	}
    }

  
}  // namespace dace


#endif  // __DACE_CUDACOPY_CUH
