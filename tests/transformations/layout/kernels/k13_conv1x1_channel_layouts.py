# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k13 1x1 convolution -- NCHW vs NHWC vs nChw16c channel layouts.

    out[n,k,h,w] = sum_c Wt[k,c] * X[n,c,h,w]

A 1x1 convolution is a per-pixel matmul over the channel axis, so the layout decision is the
ORIENTATION of the channel reduction in the activation tensor ``X``:

    NCHW    (N, C, H, W): channels strided by H*W per pixel (identity).
    NHWC    (N, H, W, C): channels contiguous per pixel -> pixel-major GEMM (a Permute of X).
    nChw16c (N, C/16, H, W, 16): 16-channel SIMD panels contiguous per pixel (a Block of X's C axis).

The Permute family is transparent (``add_permute_maps`` wraps ``X``), so every dimension order of X
reproduces the oracle; NCHW is the identity and NHWC is the permutation (0,2,3,1). The nChw16c Block
splits X's C axis -- ``SplitDimensions`` moves the 16-lane panel to the last axis, so the run closure
packs X into the physical ``[N, C/16, H, W, 16]`` layout. The reduction (not BLAS gemm) form keeps the
layout honest -- a BLAS gemm would pack its operands internally and hide the channel stride.

Source: Intel oneDNN "Understanding Memory Formats" (nChw16c); NVIDIA cuDNN (NHWC tensor-core convs);
Georganas et al., "Anatomy of High-Performance Deep Learning Convolutions on SIMD Architectures,"
SC'18 (blocked conv layouts).
"""
import numpy
import dace

from dace.transformation.layout.brute_force import permutation_candidates
from dace.transformation.layout.split_dimensions import SplitDimensions
from dace.transformation.layout.normalize_schedule import normalize_schedule_for_layout

N, C, H, W, K = (dace.symbol(s) for s in ("N", "C", "H", "W", "K"))
CBLOCK = 16  # nChw16c SIMD-lane panel width (C must be divisible by it for the Block candidate)


@dace.program
def conv1x1(X: dace.float64[N, C, H, W], Wt: dace.float64[K, C], out: dace.float64[N, K, H, W]):
    for n, k, h, w, c in dace.map[0:N, 0:K, 0:H, 0:W, 0:C] @ dace.ScheduleType.Sequential:
        out[n, k, h, w] += Wt[k, c] * X[n, c, h, w]


def oracle(X, Wt):
    return {"out": numpy.einsum("kc,nchw->nkhw", Wt, X)}


def make_inputs(n, c, h, w, k, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"X": rng.random((n, c, h, w)), "Wt": rng.random((k, c))}


def candidates():
    """Permute family over X (NCHW/NHWC/... all 24 dim orders) plus the nChw16c Block of X's C axis."""
    cands = dict(permutation_candidates("X", 4))

    def block_channels(sdfg):
        masks = [i == 1 for i in range(4)]  # block dimension 1 (the channel axis C)
        facs = [CBLOCK if i == 1 else 1 for i in range(4)]
        SplitDimensions(split_map={"X": (masks, facs)}).apply_pass(sdfg, {})
        normalize_schedule_for_layout(sdfg)

    cands[f"block_X_nChw{CBLOCK}c"] = block_channels
    return cands


def run_closure(inputs, n, c, h, w, k):
    """``run(sdfg)`` binds fresh zeroed ``out`` and reshapes X into each candidate's physical layout.

    A transparent permute keeps X's external interface at NCHW ([N,C,H,W]); the nChw16c Block gives X
    a 5-D descriptor ``[N, C/CB, H, W, CB]``, so pack the channel axis into lane-contiguous panels.
    """

    def run(sdfg):
        shp = sdfg.arrays["X"].shape
        if len(shp) == 5:  # nChw16c: (N, C/CB, H, W, CB) -- X[n,c,h,w] -> Xb[n, c//CB, h, w, c%CB]
            cb = int(shp[-1])
            X_in = inputs["X"].reshape(n, c // cb, cb, h, w).transpose(0, 1, 3, 4, 2).copy()
        else:  # transparent permute: external interface stays NCHW, pass the packed-C input as-is
            X_in = inputs["X"].copy()
        out = numpy.zeros((n, k, h, w))
        sdfg(X=X_in, Wt=inputs["Wt"].copy(), out=out, N=n, C=c, H=h, W=w, K=k)
        return {"out": out}

    return run
