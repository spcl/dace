#pragma once

#include "dace/tiles/common.h"

namespace dace_tile {

template <std::size_t M, std::size_t K, std::size_t N, typename T>
void mma_static(const T* __restrict__ A,
                const T* __restrict__ B,
                T*       __restrict__ acc,
                T*       __restrict__ C,
                std::ptrdiff_t a_row_stride = K,
                std::ptrdiff_t b_row_stride = N,
                std::ptrdiff_t acc_row_stride = N,
                std::ptrdiff_t c_row_stride = N,
                std::ptrdiff_t a_col_stride = 1,
                std::ptrdiff_t b_col_stride = 1,
                std::ptrdiff_t acc_col_stride = 1,
                std::ptrdiff_t c_col_stride = 1) noexcept
{
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            T lacc = detail::at(acc, i, j, acc_row_stride, acc_col_stride);
            for (std::size_t k = 0; k < K; ++k) {
                lacc += detail::at(A, i, k, a_row_stride, a_col_stride)
                     * detail::at(B, k, j, b_row_stride, b_col_stride);
            }
            detail::at(C, i, j, c_row_stride, c_col_stride) = lacc;
        }
    }
}

}