#pragma once

// Multiplex load for ``a[int_floor(i, D)]``: lane ``l`` reads source element ``(phase + l) / D`` from
// ``in_ptr``, where ``phase = i % D``.
template <typename T>
inline void multiplex_elements(const T* __restrict__ in_ptr, T* __restrict__ out_ptr, int W, int D, int phase) {
  for (int l = 0; l < W; ++l) {
    out_ptr[l] = in_ptr[(phase + l) / D];
  }
}
