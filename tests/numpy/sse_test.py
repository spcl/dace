# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

# Declaration of symbolic variables
Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb = (dace.symbol(name)
                                       for name in ['Nkz', 'NE', 'Nqz', 'Nw', 'N3D', 'NA', 'NB', 'Norb'])


@dace.program
def sse_sigma(neigh_idx: dace.int32[NA, NB], dH: dace.complex128[NA, NB, N3D, Norb,
                                                                 Norb], G: dace.complex128[Nkz, NE, NA, Norb, Norb],
              D: dace.complex128[Nqz, Nw, NA, NB, N3D, N3D], Sigma: dace.complex128[Nkz, NE, NA, Norb, Norb]):

    # Declaration of Map scope
    for k, E, q, w, i, j, a, b in dace.map[0:Nkz, 0:NE, 0:Nqz, 0:Nw, 0:N3D, 0:N3D, 0:NA, 0:NB]:
        # f = neigh_idx[a, b]
        dHG = G[k - q, E - w, neigh_idx[a, b]] @ dH[a, b, i]
        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
        Sigma[k, E, a] += dHG @ dHD


def test():
    sse_sigma.compile()


if __name__ == '__main__':
    test()
