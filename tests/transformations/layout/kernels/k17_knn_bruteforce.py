# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k17 brute-force k-nearest-neighbour (1-NN) -- the layout+tiling witness over an argmin reduction.

For each query point, scan every reference point and keep the nearest::

    out_idx[q]  = argmin_r  |query[q] - ref[r]|^2
    out_dist[q] = min_r     |query[q] - ref[r]|^2

The searched set ``ref`` is read in full for every query, so -- like k16 -- how ``ref`` is stored is
the layout decision. It differs from k16 in the reduction (a running-min argmin, not a sum) and in the
query/reference asymmetry.

    AoS   (NR, D) : point-major, a reference point's coords are contiguous   (identity)
    SoA   (D, NR) : coord-major, one contiguous vector per axis              (transparent Permute)

``ref[r, d]`` is read at the same logical index in every candidate, so the layout is transparent and
each reproduces the argmin oracle. Ties resolve to the first (lowest) index in both the kernel
(strict ``<``) and numpy ``argmin``. Source: brute-force kNN (the exhaustive-scan baseline behind
FLANN/FAISS IVF); SC26 layout paper (Permute over an all-pairs scan).
"""
import numpy
import dace

from dace.transformation.layout.brute_force import permutation_candidates

NR, NQ = dace.symbol("NR"), dace.symbol("NQ")
D = 3  # spatial dimension (small compile-time constant, like Norb in k10)


@dace.program
def knn(ref: dace.float64[NR, D], query: dace.float64[NQ, D], out_idx: dace.int64[NQ], out_dist: dace.float64[NQ]):
    """The q-loop is data-parallel; the r-scan is a sequential running-min argmin."""
    for q in dace.map[0:NQ] @ dace.ScheduleType.Sequential:
        best = dace.float64(1e38)
        bidx = dace.int64(0)
        for r in range(NR):
            d = dace.float64(0)
            for dm in range(D):
                diff = query[q, dm] - ref[r, dm]
                d = d + diff * diff
            if d < best:
                best = d
                bidx = r
        out_idx[q] = bidx
        out_dist[q] = best


def oracle(ref, query):
    d2 = numpy.square(query[:, None, :] - ref[None, :, :]).sum(-1)  # (NQ, NR)
    return {"out_idx": d2.argmin(1).astype(numpy.int64), "out_dist": d2.min(1)}


def make_inputs(nr, nq, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"ref": rng.random((nr, D)), "query": rng.random((nq, D))}


def candidates():
    """AoS (NR,D) identity vs SoA (D,NR): one transparent Permute of ``ref``'s two axes."""
    return dict(permutation_candidates("ref", 2))  # permute_ref_01 (aos), permute_ref_10 (soa)


def run_closure(inputs, nr, nq):
    """The Permute is transparent, so inputs bind as-is; fresh zeroed outputs each call."""

    def run(sdfg):
        out_idx = numpy.zeros(nq, dtype=numpy.int64)
        out_dist = numpy.zeros(nq)
        sdfg(ref=inputs["ref"].copy(), query=inputs["query"].copy(), out_idx=out_idx, out_dist=out_dist, NR=nr, NQ=nq)
        return {"out_idx": out_idx, "out_dist": out_dist}

    return run
