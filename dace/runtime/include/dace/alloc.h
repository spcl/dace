// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_ALLOC_H
#define __DACE_ALLOC_H

#include <cstddef>
#include <new>
#include <type_traits>

#include "types.h"

namespace dace
{
    // Aligned heap arrays. The aligned ``operator delete[]`` runs no destructors, so only trivially
    // destructible types are allocated aligned; all others use plain ``new[]`` / ``delete[]``.

    template <typename T>
    DACE_HFI T *aligned_new_array(std::size_t size, std::size_t alignment)
    {
#if defined(__cpp_aligned_new)
        // Compiler supports aligned new (C++17 feature)
        if constexpr (std::is_trivially_destructible<T>::value) {
            return new (std::align_val_t(alignment)) T[size];
        } else {
            return new T[size];
        }
#else
        // Plain new and delete[], just to be safe
        return new T[size];
#endif
    }

    template <typename T>
    DACE_HFI void aligned_delete_array(T *ptr, std::size_t alignment)
    {
#if defined(__cpp_aligned_new)
        // Compiler supports aligned new (C++17 feature)
        if constexpr (std::is_trivially_destructible<T>::value) {
            ::operator delete[](ptr, std::align_val_t(alignment));
        } else {
            delete[] ptr;
        }
#else
        // Plain new and delete[], just to be safe
        delete[] ptr;
#endif
    }
}  // namespace dace

#endif  // __DACE_ALLOC_H
