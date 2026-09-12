#pragma once

// Lane-replication ("multiplex") load for a structured-replication access
// ``a[int_floor(i, D)]``: as the vectorized index ``i`` advances by one, the
// source index advances by one only every ``D`` lanes, so ``W`` output lanes
// read ``ceil(W / D)`` distinct contiguous source elements.
//
// ``in_ptr`` points at the first distinct source element, ``a[int_floor(i, D)]``.
// ``phase = i % D`` is the base lane's offset inside its D-group: the vector
// tile starts at ``i`` (a multiple of ``W``), which need not be a multiple of
// ``D`` (e.g. D=3, W=8), so lane ``l`` reads distinct element ``(phase + l) / D``.
// When ``D`` divides ``W`` the base is a multiple of ``D`` and ``phase == 0``.
template <typename T>
inline void multiplex_elements(const T* __restrict__ in_ptr,
                               T* __restrict__ out_ptr, int W, int D, int phase) {
  for (int l = 0; l < W; ++l) {
    out_ptr[l] = in_ptr[(phase + l) / D];
  }
}
