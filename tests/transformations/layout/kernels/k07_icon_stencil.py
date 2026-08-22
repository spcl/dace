# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k07 ICON semi-structured stencil -- vertical-first vs horizontal-first as a global Permute.

A 3D field ``A[NI, NJ, NK]`` (two horizontal grid axes I, J and a vertical / level axis K) is swept
by an interior-only stencil that averages a vertical neighbour pair and the four horizontal
neighbours:

    out[i,j,k] = WV*(A[i,j,k-1] + A[i,j,k+1])
               + WH*(A[i-1,j,k] + A[i+1,j,k] + A[i,j-1,k] + A[i,j+1,k])

The layout decision is the dimension order of ``A`` -- one global Permute. In C-order the last axis
is unit-stride, so ``[0,1,2]`` (K last) is VERTICAL-first (each contiguous run is one level column, as
in the ICON (nproma, nlev, nblock) GPU order) while a permutation putting a horizontal axis last is
HORIZONTAL-first (ICON's (nproma, nlev) Fortran default). The sweep enumerates every permutation of
``A``; the permute is transparent (``add_permute_maps`` wraps the input and permutes each stencil
memlet subset), so every candidate must reproduce the oracle -- the sweep only picks the fastest.

Boundaries (i,j,k on any face) are never written and stay zero, matching the interior-only ICON
tendency stencil.

Source: G. Zaengl, D. Reinert, P. Ripodas, M. Baldauf, "The ICON modelling framework of DWD and
MPI-M," QJRMS 141, 2015 (dynamical core); M. Giorgetta et al., "The ICON-A model for direct QBO
simulations on GPUs," GMD 2022 ((nproma, nlev, nblocks) unit-stride order); SC26 layout paper SS IV-D.
"""
import numpy
import dace

from dace.transformation.layout.brute_force import permutation_candidates

NI, NJ, NK = dace.symbol("NI"), dace.symbol("NJ"), dace.symbol("NK")
WV, WH = 0.5, 0.25


@dace.program
def stencil(A: dace.float64[NI, NJ, NK], out: dace.float64[NI, NJ, NK]):
    for i, j, k in dace.map[1:NI - 1, 1:NJ - 1, 1:NK - 1] @ dace.ScheduleType.Sequential:
        out[i, j, k] = (WV * (A[i, j, k - 1] + A[i, j, k + 1]) + WH *
                        (A[i - 1, j, k] + A[i + 1, j, k] + A[i, j - 1, k] + A[i, j + 1, k]))


def oracle(A):
    out = numpy.zeros_like(A)  # boundary faces are never written -> stay 0
    out[1:-1, 1:-1, 1:-1] = (WV * (A[1:-1, 1:-1, :-2] + A[1:-1, 1:-1, 2:]) + WH *
                             (A[:-2, 1:-1, 1:-1] + A[2:, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1] + A[1:-1, 2:, 1:-1]))
    return {"out": out}


def make_inputs(ni, nj, nk, seed=0):
    return {"A": numpy.random.default_rng(seed).random((ni, nj, nk))}


def candidates():
    """The global layout candidates: every dimension permutation of ``A`` (vertical-first, etc.)."""
    return dict(permutation_candidates("A", 3))


def run_closure(inputs, ni, nj, nk):
    """A ``run(sdfg) -> outputs`` closure for the sweep. The permute is transparent (``A`` keeps its
    logical shape), so the input is bound as-is and fresh zeroed output is allocated each call."""

    def run(sdfg):
        out = numpy.zeros((ni, nj, nk))
        sdfg(A=inputs["A"].copy(), out=out, NI=ni, NJ=nj, NK=nk)
        return {"out": out}

    return run
