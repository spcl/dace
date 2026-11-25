// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Thread-block collectives of the BLOCK tile nodes: cub on CUDA, hipcub on HIP. The combine is the
// tile_apply of cuda.h, so a block reduction computes what the per-thread one does.
#pragma once

#include "cuda.h"

#if defined(__HIPCC__)
#include <hipcub/hipcub.hpp>
#elif defined(__CUDACC__)
#if __has_include(<cub/cub.cuh>)
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#else
#include "../../../../external/cub/cub/block/block_reduce.cuh"
#include "../../../../external/cub/cub/block/block_scan.cuh"
#endif
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
namespace dace {
namespace tileops {

#if defined(__HIPCC__)
namespace block_cub = hipcub;
#else
namespace block_cub = cub;
#endif

template <typename T, char Op>
struct tile_combine {
  DACE_DFI T operator()(const T& a, const T& b) const { return tile_apply<T, Op>(a, b); }
};

template <typename T, int THREADS>
using BlockReduce = block_cub::BlockReduce<T, THREADS>;

template <typename T, int THREADS>
using BlockScan = block_cub::BlockScan<T, THREADS>;

}  // namespace tileops
}  // namespace dace
#endif
