# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the ``LiftInv`` canonicalization pass.

The pass recognises a matrix inverse spelled as ``solve(A, eye(N))`` -- a
``Solve`` library node whose right-hand side is a freshly-built identity matrix
(``numpy.eye`` / ``numpy.identity``) -- and replaces the ``Solve`` plus its
identity-construction map with a single ``Inv`` node (getrf + getri). Because
``solve(A, I)`` and ``Inv`` both compute ``A^-1`` from the same LU
factorisation, the lift agrees with ``numpy.linalg.inv`` to a tight
floating-point tolerance (both are LAPACK on the same well-conditioned input).
A genuine linear solve (non-identity RHS) and a shifted / non-identity diagonal
are left untouched (opt-in, safe).

RUNTIME: the numerical checks expand ``Inv`` to LAPACK (getrf/getri via
OpenBLAS/MKL). Where that toolchain is genuinely unavailable the build raises a
``CompilationError`` and only the numerical assertion is skipped -- the
structural lift (Inv appears, Solve / identity map gone) is always checked.
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
from dace.codegen.exceptions import CompilationError, CompilerConfigurationError
from dace.libraries.linalg.nodes.inv import Inv
from dace.libraries.linalg.nodes.solve import Solve
from dace.transformation.passes.canonicalize.lift_inv import LiftInv
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol('N')


@dace.program
def solve_eye(A: dace.float64[N, N], out: dace.float64[N, N]):
    out[:] = np.linalg.solve(A, np.eye(N))


@dace.program
def solve_identity(A: dace.float64[N, N], out: dace.float64[N, N]):
    out[:] = np.linalg.solve(A, np.identity(N))


@dace.program
def real_solve(A: dace.float64[N, N], B: dace.float64[N, N], out: dace.float64[N, N]):
    """A genuine linear solve with a data right-hand side -- NOT an inverse."""
    out[:] = np.linalg.solve(A, B)


@dace.program
def shifted_diagonal(A: dace.float64[N, N], out: dace.float64[N, N]):
    """``eye(N, N, 1)`` is a shifted (super-)diagonal, not the identity."""
    out[:] = np.linalg.solve(A, np.eye(N, N, 1))


def _n_inv(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Inv))


def _n_solve(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Solve))


def _n_maps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _n_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True)
               if isinstance(r, LoopRegion) and r.loop_variable)


def _well_conditioned(n, seed):
    """A diagonally dominant (hence invertible, well-conditioned) matrix."""
    rng = np.random.default_rng(seed)
    return rng.random((n, n)) + n * np.eye(n)


def _run_or_skip(sdfg, **kwargs):
    """Run ``sdfg`` (which contains an ``Inv``); skip only if the LAPACK
    expansion genuinely cannot be built/linked in this environment."""
    try:
        sdfg(**kwargs)
    except (CompilationError, CompilerConfigurationError) as ex:
        pytest.skip(f"LAPACK (getrf/getri) unavailable to build Inv: {ex}")


def test_solve_eye_lifts_standalone():
    """``solve(A, eye(N))`` lifts to exactly one Inv node; the Solve and the
    identity-construction map are gone, and the result matches numpy.inv."""
    sdfg = solve_eye.to_sdfg(simplify=True)
    assert _n_solve(sdfg) == 1
    lifted = LiftInv().apply_pass(sdfg, {})
    assert lifted == 1, 'the solve-against-identity must lift'
    assert _n_inv(sdfg) == 1, 'exactly one Inv node must appear'
    assert _n_solve(sdfg) == 0, 'the Solve node must be gone'
    assert _n_maps(sdfg) == 0, 'the identity-construction map must be gone'
    assert _n_loops(sdfg) == 0, 'no loop should remain'
    sdfg.validate()

    n = 7
    A = _well_conditioned(n, 0)
    out = np.zeros((n, n))
    _run_or_skip(sdfg, A=A.copy(), out=out, N=n)
    assert np.allclose(out, np.linalg.inv(A), rtol=1e-9, atol=1e-11)


def test_solve_identity_lifts_standalone():
    """``numpy.identity`` is the same identity construction as ``numpy.eye`` and
    must lift identically."""
    sdfg = solve_identity.to_sdfg(simplify=True)
    lifted = LiftInv().apply_pass(sdfg, {})
    assert lifted == 1
    assert _n_inv(sdfg) == 1
    assert _n_solve(sdfg) == 0
    assert _n_maps(sdfg) == 0
    sdfg.validate()

    n = 6
    A = _well_conditioned(n, 1)
    out = np.zeros((n, n))
    _run_or_skip(sdfg, A=A.copy(), out=out, N=n)
    assert np.allclose(out, np.linalg.inv(A), rtol=1e-9, atol=1e-11)


def test_solve_eye_lifts_via_canonicalize():
    """The lift is wired into the canonicalize pipeline: running the full
    pipeline replaces ``solve(A, eye)`` with an Inv node end-to-end."""
    sdfg = solve_eye.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _n_inv(sdfg) == 1, 'canonicalize must lift the inverse to one Inv node'
    assert _n_solve(sdfg) == 0, 'the Solve node must be gone after canonicalize'
    sdfg.validate()

    n = 8
    A = _well_conditioned(n, 2)
    out = np.zeros((n, n))
    _run_or_skip(sdfg, A=A.copy(), out=out, N=n)
    assert np.allclose(out, np.linalg.inv(A), rtol=1e-9, atol=1e-11)


def test_real_solve_not_lifted():
    """A genuine linear solve (data RHS, not an identity) must NOT lift -- it is
    a real solve, not an inverse. Left as a Solve; still runs correctly."""
    sdfg = real_solve.to_sdfg(simplify=True)
    lifted = LiftInv().apply_pass(sdfg, {})
    assert not lifted, 'a non-identity RHS is a real solve, not an inverse'
    assert _n_inv(sdfg) == 0
    assert _n_solve(sdfg) == 1
    sdfg.validate()

    n = 6
    A = _well_conditioned(n, 3)
    rng = np.random.default_rng(30)
    B = rng.random((n, n))
    out = np.zeros((n, n))
    _run_or_skip(sdfg, A=A.copy(), B=B.copy(), out=out, N=n)
    assert np.allclose(out, np.linalg.solve(A, B), rtol=1e-9, atol=1e-11)


def test_shifted_diagonal_not_lifted():
    """``eye(N, N, 1)`` is a shifted diagonal, not the identity: no lift."""
    sdfg = shifted_diagonal.to_sdfg(simplify=True)
    lifted = LiftInv().apply_pass(sdfg, {})
    assert not lifted, 'a shifted diagonal is not the identity'
    assert _n_inv(sdfg) == 0
    assert _n_solve(sdfg) == 1
    sdfg.validate()


if __name__ == '__main__':
    test_solve_eye_lifts_standalone()
    test_solve_identity_lifts_standalone()
    test_solve_eye_lifts_via_canonicalize()
    test_real_solve_not_lifted()
    test_shifted_diagonal_not_lifted()
