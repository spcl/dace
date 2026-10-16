# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Assigning an expression over a promoted scalar's symbol reads a symbol the SDFG already declares.

``jk0_lo`` is a scalar; the slice bound promotes it to ``__sym_jk0_lo_0``, which the slice's shape carries. Reading
``part.shape[1]`` assigns that shape, and the frontend looked the symbol up among the program's names to type it --
where a promoted symbol is not -- before seeing the SDFG declares it already (velocity_tendencies).
"""

import numpy as np

import dace

N, M, nlev, nrdmax = (dace.symbol(s) for s in ("N", "M", "nlev", "nrdmax"))


@dace.program
def last_level_of_a_promoted_slice(a: dace.float64[N, nlev, M], out: dace.float64[N]):
    jk0_lo = max(3, nrdmax - 2) - 1
    if nlev - 4 >= jk0_lo:
        part = np.abs(a[:, jk0_lo : nlev - 4 + 1, :])
        d1 = part.shape[1]
        for i in range(N):
            out[i] = part[i, d1 - 1, 0]


def test_a_shape_over_a_promoted_scalar_symbol_assigns():
    sdfg = last_level_of_a_promoted_slice.to_sdfg(simplify=True)
    a = np.random.default_rng(0).random((5, 12, 3))
    out = np.zeros(5)
    sdfg(a=a, out=out, N=5, M=3, nlev=12, nrdmax=6)
    assert np.allclose(out, np.abs(a[:, 12 - 4, 0]))


if __name__ == "__main__":
    test_a_shape_over_a_promoted_scalar_symbol_assigns()
