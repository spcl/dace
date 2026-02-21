#pragma once

#include "dace/tiles/common.h"

namespace dace_tile {

template <std::size_t ROWS, std::size_t COLS, typename T>
void store_static(const T* __restrict__ src,
                  T*       __restrict__ dst,
                  std::ptrdiff_t        dst_row_stride,
                  std::ptrdiff_t        dst_col_stride = 1,
                  std::ptrdiff_t        src_row_stride = COLS,
                  std::ptrdiff_t        src_col_stride = 1) noexcept
{
    for (std::size_t i = 0; i < ROWS; ++i)
        for (std::size_t j = 0; j < COLS; ++j)
            detail::at(dst, i, j, dst_row_stride, dst_col_stride) =
                detail::at(src, i, j, src_row_stride, src_col_stride);
}

}