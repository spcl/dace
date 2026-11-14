// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// The device strided scan's plan: which of ``dace/cuda/scan.cuh``'s paths runs a strided scan of stride ``s``.
// Host-only, so the rule compiles and is testable without a device toolchain.

#ifndef __DACE_CUDA_SCAN_PLAN_H
#define __DACE_CUDA_SCAN_PLAN_H

namespace dace {
namespace cuda_scan {
namespace plan {

//: Threads per block of the tiled path.
constexpr int kTileThreads = 256;
//: Classes per tile and row groups per block of the tiled strided path: a warp-wide run of classes, so every
//: row group's load of one row is a contiguous segment.
constexpr int kTileClasses = 32;
constexpr int kTileRowGroups = kTileThreads / kTileClasses;
//: Rows one thread walks at least, and the block count the tiled path splits rows to reach.
constexpr long kTileMinRowsPerThread = 16;
constexpr long kTileTargetBlocks = 4096;
//: Below this many rows one launch walks them all: two more launches cost more than the shorter walk saves.
constexpr long kTileOneChunkBelow = 512;
//: From this many classes the tiled path gets at most two row chunks, and one thread per class does better.
constexpr long kPerClassAtLeast = kTileClasses * kTileTargetBlocks / 2;
//: Below this stride (SMALL, two warps of classes) a block scanning each class's chunks beats the tiles.
constexpr long kTiledAtLeast = 2 * kTileClasses;

//: How a strided scan runs on the device.
enum class StridedPlan { BLOCKED, TILED, PER_CLASS };

//: Below ``kTiledAtLeast`` classes a block scans one class's chunk (BLOCKED); from ``kPerClassAtLeast`` classes one
//: thread walks each class (PER_CLASS); in between class tiles by row chunks (TILED), whose row groups keep a short
//: chain's walk short. A SMALL stride is never split across lanes: it is BLOCKED.
inline StridedPlan choose_strided_plan(long s) {
  if (s < kTiledAtLeast) return StridedPlan::BLOCKED;
  if (s >= kPerClassAtLeast) return StridedPlan::PER_CLASS;
  return StridedPlan::TILED;
}

}  // namespace plan
}  // namespace cuda_scan
}  // namespace dace

#endif  // __DACE_CUDA_SCAN_PLAN_H
