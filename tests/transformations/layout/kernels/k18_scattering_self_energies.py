# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k18 scattering self-energies (npbench ``sselfeng``) -- k10's cross-phase Permute WITH an indirect
neighbour gather.

The OMEN electron-phonon scattering self-energy: ``Sigma`` accumulates, over phonon momenta/energies
``(q, w)`` and each atom's neighbours ``b``, a product of Green's-function and coupling blocks::

    Sigma[k, E, a] += ( G[k, E-w, neigh_idx[a,b]] @ dH[a,b,i] ) @ ( dH[a,b,j] * D[q,w,a,b,i,j] )

(for every ``E-w >= 0``). The layout-critical tensor is ``G`` -- a stack of tiny ``Norb x Norb`` blocks
over batch axes ``(Nkz, NE, NA)``. Like k10 the storage decision is the ORDER of those batch axes,
but here the atom axis ``NA`` is read through the data-dependent neighbour index ``neigh_idx[a,b]``:

    energy-outer  (Nkz, NE, NA, .) : identity -- a whole-atom energy slab is strided.
    atom-outer    (Nkz, NA, NE, .) : each atom's energy spectrum contiguous, so the ``E-w`` window
                                     over a gathered neighbour atom is a contiguous read (Permute).

``G`` is an input, and the Permute is transparent (``add_permute_maps`` rewrites every read of ``G``,
including the indirect gather), so both candidates reproduce the oracle; the sweep only picks the
physical order. This is the single-process form of the distributed OMEN transpose in
``tests/library/mpi/mpi_omen_transpose_test.py`` -- the same batch-axis reorientation, here local and
with the neighbour gather k10 abstracts away.

Source: A. N. Ziogas et al., "A Data-Centric Approach to Extreme-Scale Ab initio Dissipative Quantum
Transport Simulations," SC'19 (Gordon Bell finalist; arXiv:1912.10024); npbench ``scattering_self_energies``
(dense_linear_algebra); SC26 layout paper.
"""
import numpy
import dace

from dace.transformation.layout.permute_dimensions import PermuteDimensions

NA, NE = dace.symbol("NA"), dace.symbol("NE")
Nkz, NB, Nqz, Nw, Norb, N3D = 2, 2, 2, 2, 3, 2  # small compile-time block/momentum dims (npbench 'S' preset)


@dace.program
def sselfeng(neigh_idx: dace.int32[NA, NB], dH: dace.complex128[NA, NB, N3D, Norb,
                                                                Norb], G: dace.complex128[Nkz, NE, NA, Norb, Norb],
             D: dace.complex128[Nqz, Nw, NA, NB, N3D, N3D], Sigma: dace.complex128[Nkz, NE, NA, Norb, Norb]):
    for k in range(Nkz):
        for E in range(NE):
            for q in range(Nqz):
                for w in range(Nw):
                    for i in range(N3D):
                        for j in range(N3D):
                            for a in range(NA):
                                for b in range(NB):
                                    if E - w >= 0:
                                        dHG = G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i]
                                        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                        Sigma[k, E, a] += dHG @ dHD


def oracle(neigh_idx, dH, G, D):
    Sigma = numpy.zeros([Nkz, G.shape[1], G.shape[2], Norb, Norb], dtype=numpy.complex128)
    for k in range(G.shape[0]):
        for E in range(G.shape[1]):
            for q in range(D.shape[0]):
                for w in range(D.shape[1]):
                    for i in range(D.shape[-2]):
                        for j in range(D.shape[-1]):
                            for a in range(neigh_idx.shape[0]):
                                for b in range(neigh_idx.shape[1]):
                                    if E - w >= 0:
                                        dHG = G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i]
                                        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                        Sigma[k, E, a] += dHG @ dHD
    return {"Sigma": Sigma}


def make_inputs(na, ne, seed=0):
    rng = numpy.random.default_rng(seed)
    cplx = lambda *s: (rng.random(s) + 1j * rng.random(s)).astype(numpy.complex128)
    neigh_idx = numpy.zeros([na, NB], dtype=numpy.int32)
    for i in range(na):
        neigh_idx[i] = numpy.mod(numpy.arange(i - NB // 2, i - NB // 2 + NB), na)  # each atom's NB neighbours
    return {
        "neigh_idx": neigh_idx,
        "dH": cplx(na, NB, N3D, Norb, Norb),
        "G": cplx(Nkz, ne, na, Norb, Norb),
        "D": cplx(Nqz, Nw, na, NB, N3D, N3D),
    }


def candidates():
    """The batch-axis order of ``G``: energy-outer (identity) vs atom-outer (swap NE and NA, so the
    gathered neighbour's energy window is contiguous) -- one transparent Permute, the k10 decision."""

    def atom_outer(sdfg):
        PermuteDimensions(permute_map={"G": [0, 2, 1, 3, 4]}, add_permute_maps=True).apply_pass(sdfg, {})

    return {"e_outer": (lambda sdfg: None), "atom_outer": atom_outer}


def run_closure(inputs, na, ne):
    """``G``'s Permute is transparent -> inputs bind as-is; a fresh zeroed ``Sigma`` each call."""

    def run(sdfg):
        Sigma = numpy.zeros([Nkz, ne, na, Norb, Norb], dtype=numpy.complex128)
        sdfg(neigh_idx=inputs["neigh_idx"].copy(),
             dH=inputs["dH"].copy(),
             G=inputs["G"].copy(),
             D=inputs["D"].copy(),
             Sigma=Sigma,
             NA=na,
             NE=ne)
        return {"Sigma": Sigma}

    return run
