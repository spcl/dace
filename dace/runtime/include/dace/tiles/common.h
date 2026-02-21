#pragma once

#include <cstddef>      // std::size_t, std::ptrdiff_t
#include <stdexcept>    // std::invalid_argument  (runtime guards)

namespace dace_tile {

namespace detail {

template <typename T>
inline T& at(T* __restrict__ ptr,
             std::ptrdiff_t row, std::ptrdiff_t col,
             std::ptrdiff_t row_stride,
             std::ptrdiff_t col_stride = 1) noexcept
{
    return ptr[row * row_stride + col * col_stride];
}

template <typename T>
inline const T& at(const T* __restrict__ ptr,
                   std::ptrdiff_t row, std::ptrdiff_t col,
                   std::ptrdiff_t row_stride,
                   std::ptrdiff_t col_stride = 1) noexcept
{
    return ptr[row * row_stride + col * col_stride];
}

} // namespace detail

} // namespace dace_tile