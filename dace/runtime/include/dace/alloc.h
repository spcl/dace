// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cstdlib>

namespace dace {

// Allocate `count` elements of type T aligned to `alignment` bytes. The byte
// size is rounded up to a multiple of the alignment, as std::aligned_alloc
// requires. Free with dace::free.
template <typename T>
inline T *aligned_alloc(size_t count, size_t alignment = 64) {
    const size_t bytes = ((sizeof(T) * count + alignment - 1) / alignment) * alignment;
    return static_cast<T *>(std::aligned_alloc(alignment, bytes));
}

// Allocate and zero-initialize `count` elements of type T. Free with dace::free.
template <typename T>
inline T *calloc(size_t count) {
    return static_cast<T *>(std::calloc(count, sizeof(T)));
}

// Free memory obtained from dace::aligned_alloc or dace::calloc.
inline void free(void *ptr) {
    std::free(ptr);
}

}  // namespace dace
