// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Thread-block collectives of the BLOCK tile nodes: cub on CUDA, hipcub on HIP (gpucub.cuh). The combine is
// the tile_apply of cuda.h, so a block reduction computes what the per-thread one does.
#pragma once

#include "cuda.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#include "../cuda/gpucub.cuh"

namespace dace {
namespace tileops {

template <typename T, char Op>
struct tile_combine {
  DACE_DFI T operator()(const T& a, const T& b) const { return tile_apply<T, Op>(a, b); }
};

template <typename T, int THREADS>
using BlockReduce = gpucub::BlockReduce<T, THREADS>;

template <typename T, int THREADS>
using BlockScan = gpucub::BlockScan<T, THREADS>;

}  // namespace tileops
}  // namespace dace
#endif
