# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``scattering_self_energies`` (dense_linear_algebra) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float32
dc_complex_float = dc.complex64


def rng_complex(shape, rng, datatype):
    return (rng.random(shape, dtype=datatype) + rng.random(shape, dtype=datatype) * 1j)


SIZES = {'Nkz': 2, 'NE': 4, 'Nqz': 2, 'Nw': 2, 'N3D': 2, 'NA': 6, 'NB': 2, 'Norb': 3}
INPUT_ARGS = ('Nkz', 'NE', 'Nqz', 'Nw', 'N3D', 'NA', 'NB', 'Norb')
ARRAY_ARGS = ('neigh_idx', 'dH', 'G', 'D', 'Sigma')
SCALARS = {}
OUTPUT_ARGS = ('Sigma', )

NA, NB, Nkz, NE, Nqz, Nw, Norb, N3D = (dc.symbol(s, dc.int64)
                                       for s in ('NA', 'NB', 'Nkz', 'NE', 'Nqz', 'Nw', 'Norb', 'N3D'))


def initialize(Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb, datatype=np.float32):
    from numpy.random import default_rng
    rng = default_rng(42)
    neigh_idx = np.ndarray([NA, NB], dtype=np.int32)
    for i in range(NA):
        neigh_idx[i] = np.positive(np.arange(i - NB / 2, i + NB / 2) % NA)
    dH = rng_complex([NA, NB, N3D, Norb, Norb], rng, datatype)
    G = rng_complex([Nkz, NE, NA, Norb, Norb], rng, datatype)
    D = rng_complex([Nqz, Nw, NA, NB, N3D, N3D], rng, datatype)
    Sigma = np.zeros([Nkz, NE, NA, Norb, Norb], dtype=D.dtype)
    return (neigh_idx, dH, G, D, Sigma)


def reference(neigh_idx, dH, G, D, Sigma):
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


@dc.program
def kernel(neigh_idx: dc.int32[NA, NB], dH: dc_complex_float[NA, NB, N3D, Norb, Norb], G: dc_complex_float[Nkz, NE, NA,
                                                                                                           Norb, Norb],
           D: dc_complex_float[Nqz, Nw, NA, NB, N3D, N3D], Sigma: dc_complex_float[Nkz, NE, NA, Norb, Norb]):
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


CORPUS = dict(name='scattering_self_energies',
              dwarf='dense_linear_algebra',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
