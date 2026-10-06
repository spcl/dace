// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_ALLOC_H
#define __DACE_ALLOC_H

#include <cstddef>
#include <new>
#include <type_traits>

namespace dace
{
#ifdef __cpp_aligned_new
    // Aligned heap arrays. The aligned ``operator delete[]`` runs no destructors, so only trivially
    // destructible types are allocated aligned; all others use plain ``new[]`` / ``delete[]``.

    template <typename T>
    T *aligned_new_array(std::size_t size, std::size_t alignment)
    {
        if constexpr (std::is_trivially_destructible<T>::value) {
            return new (std::align_val_t(alignment)) T[size];
        } else {
            return new T[size];
        }
    }

    template <typename T>
    void aligned_delete_array(T *ptr, std::size_t alignment)
    {
        if constexpr (std::is_trivially_destructible<T>::value) {
            ::operator delete[](ptr, std::align_val_t(alignment));
        } else {
            delete[] ptr;
        }
    }
#endif
}  // namespace dace

#endif  // __DACE_ALLOC_H
