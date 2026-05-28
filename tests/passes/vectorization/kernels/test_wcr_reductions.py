# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""WCR-reduction kernels: a map whose body writes to a scalar accumulator
with ``wcr`` (a parallel reduction). After ``NormalizeWCRSource`` runs the
WCR-source is an :class:`~dace.nodes.AccessNode` (private scalar transient
interposed between the body's CodeNode and the outer MapExit / sink), which
is the precondition for DaCe cpu codegen to emit the WCR as an OpenMP
reduction. The vectorisation path will eventually emit a per-lane tile
horizontal-reduce into that private scalar so the outer WCR aggregates
across worker chunks.

These tests pin the **numerical contract** end-to-end against the
unvectorised reference; the structural contract (NSDFG-sourced WCR -> private
scalar) is exercised by ``tests/passes/vectorization/passes`` audits."""
import dace
import numpy
import pytest

from tests.passes.vectorization.helpers.harness import N, run_vectorization_test

# WCR-reduction kernels exercise the K-dim tile-op config alongside the rest.
pytestmark = pytest.mark.tile_nodes


@dace.program
def sum_reduce(a: dace.float64[N], acc: dace.float64[1]):
    """Element-wise sum reduction. The body writes to ``acc`` via WCR so the
    outer map can parallelise — DaCe lowers ``acc`` as an OpenMP reduction
    when the WCR source is an AccessNode."""
    acc[0] = 0.0
    for i in dace.map[0:N]:
        acc[0] += a[i]


@dace.program
def max_reduce(a: dace.float64[N], m: dace.float64[1]):
    """Max reduction via WCR ``lambda a, b: max(a, b)`` from the frontend."""
    m[0] = -1.0e300
    for i in dace.map[0:N]:
        m[0] = max(m[0], a[i])


def _skip_if_nested_arm():
    """Skip on the ``--tile-nest-bodies`` arm.

    The augassign frontend lowering inside an outer map body produces a
    ``tmp`` Scalar transient whose data descriptor lives on the OUTER SDFG
    arrays dict; ``nest_state_subgraph`` (run from
    :class:`NestInnermostMapBodyIntoNSDFG` on the nested arm) enumerates
    every data name referenced inside the subgraph and trips on ``tmp``
    because the descriptor isn't on the inner SDFG yet. Pre-existing
    interaction with the augassign frontend lowering; tracked separately
    and tested on the default arm (where the body stays flat).
    """
    from tests.passes.vectorization.helpers import harness as _harness
    if _harness.FORCE_NEST_MAP_BODIES:
        pytest.skip("nest_state_subgraph cannot recover ``tmp`` augassign transient inside an inner map body")


def test_sum_reduce():
    """Numerical correctness of a sum reduction with WCR."""
    _skip_if_nested_arm()
    _N = 64
    a = numpy.random.random(_N)
    acc = numpy.zeros(1, dtype=numpy.float64)
    run_vectorization_test(
        dace_func=sum_reduce,
        arrays={"a": a, "acc": acc},
        params={"N": _N},
        sdfg_name="sum_reduce",
    )


def test_max_reduce():
    """Numerical correctness of a max reduction with WCR."""
    _skip_if_nested_arm()
    _N = 64
    a = numpy.random.random(_N)
    m = numpy.zeros(1, dtype=numpy.float64)
    run_vectorization_test(
        dace_func=max_reduce,
        arrays={"a": a, "m": m},
        params={"N": _N},
        sdfg_name="max_reduce",
    )
