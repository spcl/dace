// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// CUDA_WARP addendum: the only op a warp needs that a single lane cannot express.
// Everything else is reused from cuda.h at VLEN=R, a lane's fragment count.
#pragma once

#include "cuda.h"

namespace dace {
namespace tileops {

#if defined(__CUDACC__)
// Combines one partial per lane (typically tile_reduce<T, R, Op>'s result) across
// the warp; every lane ends holding the total. ``W`` is the tile's lane count: a
// compile-time property of the decomposition, not the runtime ``warpSize``, so the
// butterfly unrolls and the participating mask is derived rather than assumed. Op
// follows tile_reduce ('+', '*', 'm', 'M'), applied in native T (fp64 included).
template <typename T, char Op, int W = 32>
DACE_DFI T tile_reduce_warp(T partial) {
  static_assert(W > 0 && (W & (W - 1)) == 0 && W <= 32, "W must be a power of two lane count within a warp");
  constexpr unsigned mask = (W == 32) ? 0xffffffffu : ((1u << W) - 1u);
#pragma unroll
  for (int off = W / 2; off > 0; off >>= 1) {
    const T other = __shfl_xor_sync(mask, partial, off);
    if constexpr (Op == '+')
      partial = partial + other;
    else if constexpr (Op == '*')
      partial = partial * other;
    else if constexpr (Op == 'm')
      partial = partial < other ? partial : other;
    else /* 'M' */
      partial = partial > other ? partial : other;
  }
  return partial;
}
#endif

}  // namespace tileops
}  // namespace dace
