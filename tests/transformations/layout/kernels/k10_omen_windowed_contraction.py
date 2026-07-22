# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k10 OMEN SSE windowed batched contraction -- the cross-phase Permute witness (complex128).

Two phases share a tensor ``G`` of small ``M x M`` complex blocks indexed by an atom axis ``NA`` and
an energy axis ``NE``:

    producer (RGF-like):     G[a, e]     = H[a, e] @ X[a, e]
    consumer (SSE window):   Sigma[a, eo] = sum_{w<W} G[a, eo + W-1-w] @ D[w]

The layout decision is the dimension ORDER of ``G``'s two batch axes -- one global Permute:

    consumer-preferred (NA, NE, M, M): a per-atom energy window ``G[a]`` is one contiguous slab
                                       (cache-resident across the W window passes).
    producer-preferred (NE, NA, M, M): a per-energy all-atom slab is contiguous for the producer,
                                       but the consumer's window becomes large-stride.

``G`` is an internal transient, so permuting it is fully transparent (``add_permute_maps`` permutes
every producer write and consumer read memlet); every candidate reproduces the oracle and the sweep
only picks the physical order. ``M`` (~ Norb orbitals) and the window ``W`` are small compile-time
constants -- the ``NA``/``NE`` batch dimensions carry the bandwidth.

Source: A. N. Ziogas, T. Ben-Nun, G. Indalecio Fernandez, T. Schulthess, T. Hoefler, "A Data-Centric
Approach to Extreme-Scale Ab initio Dissipative Quantum Transport Simulations," SC 2019 (Gordon Bell
finalist; arXiv:1912.10024); NPBench 'sselfeng' (Ziogas et al., ICS'21); SC26 layout paper.
"""
import numpy
import dace

from dace.transformation.layout.permute_dimensions import PermuteDimensions

NA, NE = dace.symbol("NA"), dace.symbol("NE")
M = 4   # ~ Norb orbitals per block (small: a block spans a few cache lines)
W = 3   # SSE energy window width


@dace.program
def omen(H: dace.complex128[NA, NE, M, M], X: dace.complex128[NA, NE, M, M], D: dace.complex128[W, M, M],
         Sigma: dace.complex128[NA, NE - W, M, M]):
    G = numpy.empty((NA, NE, M, M), dace.complex128)  # single-writer (scalar-acc) so Permute(G) is legal
    for a, e, i, j in dace.map[0:NA, 0:NE, 0:M, 0:M] @ dace.ScheduleType.Sequential:
        s = dace.complex128(0)
        for k in range(M):
            s = s + H[a, e, i, k] * X[a, e, k, j]
        G[a, e, i, j] = s
    for a, eo, i, j, w, k in dace.map[0:NA, 0:NE - W, 0:M, 0:M, 0:W, 0:M] @ dace.ScheduleType.Sequential:
        Sigma[a, eo, i, j] += G[a, eo + W - 1 - w, i, k] * D[w, k, j]


def oracle(H, X, D):
    G = numpy.matmul(H, X)  # (NA, NE, M, M)
    na, ne = G.shape[0], G.shape[1]
    eout = ne - W
    Sigma = numpy.zeros((na, eout, M, M), dtype=numpy.complex128)
    for a in range(na):
        for eo in range(eout):
            acc = numpy.zeros((M, M), dtype=numpy.complex128)
            for w in range(W):
                acc = acc + G[a, eo + W - 1 - w] @ D[w]
            Sigma[a, eo] = acc
    return {"Sigma": Sigma}


def make_inputs(na, ne, seed=0):
    rng = numpy.random.default_rng(seed)
    cplx = lambda *s: (rng.random(s) + 1j * rng.random(s)).astype(numpy.complex128)
    return {"H": cplx(na, ne, M, M), "X": cplx(na, ne, M, M), "D": cplx(W, M, M)}


def candidates():
    """The global layout candidates: the dimension order of ``G``'s two batch axes -- consumer
    order (NA, NE) identity vs producer order (NE, NA) via a transparent Permute."""

    def producer_order(sdfg):
        PermuteDimensions(permute_map={"G": [1, 0, 2, 3]}, add_permute_maps=True).apply_pass(sdfg, {})

    return {"ae_consumer": (lambda sdfg: None), "ea_producer": producer_order}


def run_closure(inputs, na, ne):
    """``G`` is internal, so the layout is transparent -- inputs bind as-is and a fresh zeroed
    ``Sigma`` (shape ``(NA, NE-W, M, M)``) is allocated each call."""

    def run(sdfg):
        Sigma = numpy.zeros((na, ne - W, M, M), dtype=numpy.complex128)
        sdfg(H=inputs["H"].copy(), X=inputs["X"].copy(), D=inputs["D"].copy(), Sigma=Sigma, NA=na, NE=ne)
        return {"Sigma": Sigma}

    return run
