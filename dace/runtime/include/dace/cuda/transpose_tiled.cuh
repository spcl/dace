// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Tiled transpose and in-place symmetrize.
//
// A transpose reads along rows and writes along columns, so one of the two accesses is strided
// whichever way it is written: the naive kernel issues a separate memory transaction per element on
// the strided side. Staging a square tile through shared memory makes BOTH sides run along rows.
//
// The tile is declared ``[TILE][TILE + 1]``. That padding column is what removes the shared-memory
// bank conflict: with a plain ``[TILE][TILE]`` the transposed read walks a column, whose entries are
// TILE apart and therefore all in the same bank, serializing the warp 32 ways.

#pragma once

#include "cudacommon.cuh"  // the backend runtime header, plus the gpu* aliases used below

namespace dace {
namespace cuda_transpose {

//: One tile edge. 32 makes a tile row exactly one warp, so a tile's global read and its global
//: write are each a single coalesced burst.
constexpr int TILE = 32;
//: Tile rows a block covers per step; each thread walks ``TILE / BLOCK_ROWS`` elements. 8 keeps the
//: block at 256 threads, which is enough to hide the latency without starving occupancy.
constexpr int BLOCK_ROWS = 8;

inline int tiles_along(int extent) { return (extent + TILE - 1) / TILE; }

/// ``out[c, r] = in[r, c]`` for an ``rows x cols`` input. Row-major, leading dimensions given.
template <typename T>
__global__ void transpose_kernel(const T *__restrict__ in, T *__restrict__ out, int rows, int cols, int ld_in,
                                 int ld_out) {
    __shared__ T tile[TILE][TILE + 1];

    int c = blockIdx.x * TILE + threadIdx.x;
    int r = blockIdx.y * TILE + threadIdx.y;
    for (int k = 0; k < TILE; k += BLOCK_ROWS) {
        if (c < cols && (r + k) < rows) {
            tile[threadIdx.y + k][threadIdx.x] = in[(long long)(r + k) * ld_in + c];
        }
    }
    __syncthreads();

    // Swap the roles of the block indices so the WRITE also runs along a row.
    c = blockIdx.y * TILE + threadIdx.x;
    r = blockIdx.x * TILE + threadIdx.y;
    for (int k = 0; k < TILE; k += BLOCK_ROWS) {
        if (c < rows && (r + k) < cols) {
            out[(long long)(r + k) * ld_out + c] = tile[threadIdx.x][threadIdx.y + k];
        }
    }
}

/// Mirror one triangle of an ``n x n`` matrix into the other, in place.
///
/// ``source_upper``: read ``x[i, j]`` with ``j >= i + col_offset`` and write ``x[j, i]``; otherwise
/// read the lower triangle and write the upper. Blocks on the destination side of the diagonal exit
/// immediately -- they hold no source element.
///
/// Every destination address is written by exactly one thread of one block, so the in-place update
/// needs no ordering between blocks: the source triangle is never a destination.
template <typename T>
__global__ void symmetrize_kernel(T *__restrict__ x, int n, int ld, int col_offset, int source_upper) {
    __shared__ T tile[TILE][TILE + 1];

    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    if (source_upper ? (tile_col < tile_row) : (tile_col > tile_row)) return;

    const int r0 = tile_row * TILE;
    const int c0 = tile_col * TILE;
    for (int k = 0; k < TILE; k += BLOCK_ROWS) {
        const int r = r0 + threadIdx.y + k;
        const int c = c0 + threadIdx.x;
        if (r < n && c < n) {
            tile[threadIdx.y + k][threadIdx.x] = x[(long long)r * ld + c];
        }
    }
    __syncthreads();

    for (int k = 0; k < TILE; k += BLOCK_ROWS) {
        // ``tile[tx][ty + k]`` holds the source element (i, j) below; the write goes to (j, i), whose
        // column index runs with threadIdx.x and is therefore coalesced.
        const int i = r0 + threadIdx.x;
        const int j = c0 + threadIdx.y + k;
        if (i >= n || j >= n) continue;
        const bool is_source = source_upper ? (j >= i + col_offset) : (i >= j + col_offset);
        if (!is_source) continue;
        x[(long long)j * ld + i] = tile[threadIdx.x][threadIdx.y + k];
    }
}

template <typename T>
gpuError_t transpose(const T *in, T *out, int rows, int cols, int ld_in, int ld_out, gpuStream_t stream) {
    if (rows <= 0 || cols <= 0) return gpuSuccess;
    const dim3 grid(tiles_along(cols), tiles_along(rows), 1);
    const dim3 block(TILE, BLOCK_ROWS, 1);
    transpose_kernel<T><<<grid, block, 0, stream>>>(in, out, rows, cols, ld_in, ld_out);
    return gpuPeekAtLastError();
}

template <typename T>
gpuError_t symmetrize(T *x, int n, int ld, int col_offset, bool source_upper, gpuStream_t stream) {
    if (n <= 0) return gpuSuccess;
    const int nt = tiles_along(n);
    const dim3 grid(nt, nt, 1);
    const dim3 block(TILE, BLOCK_ROWS, 1);
    symmetrize_kernel<T><<<grid, block, 0, stream>>>(x, n, ld, col_offset, source_upper ? 1 : 0);
    return gpuPeekAtLastError();
}

}  // namespace cuda_transpose
}  // namespace dace
