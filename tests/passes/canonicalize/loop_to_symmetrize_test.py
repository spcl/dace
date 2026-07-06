# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the ``Symmetrize`` library node and the ``LoopToSymmetrize``
canonicalization pass.

The pass lifts a triangular in-place matrix-symmetrization loop nest
(``for i: for j in i+1:M: X[j,i] = X[i,j]``) to a ``Symmetrize`` node whose pure
expansion is a parallel triangular copy -- turning a nest that ``LoopToMap``
refuses (in-place symmetric read/write false-dependence) into a fully parallel
form.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.libraries.standard.nodes import Symmetrize
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.loop_to_symmetrize import LoopToSymmetrize

M = dace.symbol('M')


@dace.program
def symmetrize_upper(X: dace.float64[M, M]):
    for i in range(0, M - 1):
        for j in range(i + 1, M):
            X[j, i] = X[i, j]


@dace.program
def symmetrize_lower(X: dace.float64[M, M]):
    for i in range(0, M - 1):
        for j in range(i + 1, M):
            X[i, j] = X[j, i]


@dace.program
def not_symmetrize(A: dace.float64[M, M], B: dace.float64[M, M]):
    """Cross-array copy (not in-place) -- must NOT lift to Symmetrize."""
    for i in range(0, M - 1):
        for j in range(i + 1, M):
            A[j, i] = B[i, j]


def _nsym(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Symmetrize))


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion) and r.loop_variable)


def _mirror(X, source_upper):
    exp = X.copy()
    m = X.shape[0]
    for i in range(m):
        for j in range(i + 1, m):
            if source_upper:
                exp[j, i] = exp[i, j]
            else:
                exp[i, j] = exp[j, i]
    return exp


def test_node_expands_and_runs():
    """A Symmetrize node builds, expands to a parallel triangular copy, and runs."""
    sdfg = dace.SDFG('sym_node')
    sdfg.add_array('X', [M, M], dace.float64)
    st = sdfg.add_state()
    node = Symmetrize('sym', row_lo='0', row_hi='M', col_offset=1, col_hi='M', source_upper=True)
    st.add_node(node)
    st.add_edge(st.add_read('X'), None, node, '_in', dace.Memlet('X[0:M, 0:M]'))
    st.add_edge(node, '_out', st.add_write('X'), None, dace.Memlet('X[0:M, 0:M]'))
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    m = 7
    rng = np.random.default_rng(0)
    X = rng.standard_normal((m, m))
    got = X.copy()
    sdfg(X=got, M=m)
    assert np.allclose(got, _mirror(X, True))
    assert np.allclose(got, got.T)


@pytest.mark.parametrize('prog,source_upper', [(symmetrize_upper, True), (symmetrize_lower, False)])
def test_lifts_and_parallelizes(prog, source_upper):
    """The triangular symmetrization nest lifts to one Symmetrize node, leaves no
    sequential loop, and stays value-correct."""
    sdfg = prog.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nsym(sdfg) == 1, 'the symmetrization nest must lift to exactly one Symmetrize node'
    assert _nloops(sdfg) == 0, 'no sequential loop should remain'
    sdfg.validate()

    m = 9
    rng = np.random.default_rng(1)
    X = rng.standard_normal((m, m))
    got = X.copy()
    sdfg(X=got, M=m)
    assert np.allclose(got, _mirror(X, source_upper))


def test_cross_array_copy_not_lifted():
    """A cross-array (not in-place) triangular copy must NOT lift to Symmetrize --
    it is already a false-dependence-free parallel copy. Canonicalize leaves no
    Symmetrize node and stays value-correct."""
    sdfg = not_symmetrize.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nsym(sdfg) == 0

    m = 8
    rng = np.random.default_rng(2)
    A0, B = rng.standard_normal((m, m)), rng.standard_normal((m, m))
    got = A0.copy()
    sdfg(A=got, B=B.copy(), M=m)
    exp = A0.copy()
    for i in range(m):
        for j in range(i + 1, m):
            exp[j, i] = B[i, j]
    assert np.allclose(got, exp)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
