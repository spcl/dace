# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A BLAS call over an extent that is a power of a symbol (ls3df_scf's ``Lb**3``) is emitted as C++, not Python."""

import numpy as np
import pytest

import dace
from dace.libraries.blas.environments import openblas
from dace.sdfg import nodes

Lb = dace.symbol("Lb")
P = dace.symbol("P")


@dace.program
def transpose_and_dot(a: dace.float64[Lb**3, P], b: dace.float64[P, Lb**3], x: dace.float64[Lb**3], r: dace.float64[1]):
    b[:] = np.transpose(a)
    r[0] = np.dot(x, x)


@pytest.mark.skipif(not openblas.OpenBLAS.is_installed(), reason="needs OpenBLAS")
def test_blas_extents_are_cpp():
    sdfg = transpose_and_dot.to_sdfg()
    for node, state in list(sdfg.all_nodes_recursive()):
        if isinstance(node, nodes.LibraryNode):
            node.implementation = "OpenBLAS"
    sdfg.expand_library_nodes()
    calls = [
        n.code.as_string
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.Tasklet) and "cblas_" in n.code.as_string
    ]
    assert len(calls) == 2 and not any("**" in c for c in calls), calls

    a = np.random.default_rng(0).random((8, 3))
    b, x, r = np.zeros((3, 8)), np.arange(8.0), np.zeros(1)
    sdfg(a=a, b=b, x=x, r=r, Lb=2, P=3)
    assert np.allclose(b, a.T) and np.isclose(r[0], x @ x)


if __name__ == "__main__":
    test_blas_extents_are_cpp()
