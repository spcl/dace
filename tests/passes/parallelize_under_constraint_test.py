# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ParallelizeUnderConstraint: guard-and-parallelize a symbolic-stride loop.

TSVC ``s171`` (``a[i*inc] = a[i*inc] + b[i]``) is data-parallel iff ``inc != 0``.
The pass should emit a runtime ``__builtin_trap()`` guard that fires on the
violated assumption (``inc == 0``) and lift the loop to a plain (WCR-free) Map.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


@dace.program
def affine_stride_rmw(a: dace.float64[N], b: dace.float64[N], inc: dace.int64):
    for i in range(N):
        a[i * inc] = a[i * inc] + b[i]


def _trap_tasklets(sdfg):
    return [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.Tasklet) and '__builtin_trap' in (n.code.as_string or '')
    ]


def test_symbolic_stride_emits_trap_guard_and_parallel_map():
    sdfg = affine_stride_rmw.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)

    traps = _trap_tasklets(sdfg)
    assert len(traps) == 1, f'expected exactly one runtime trap guard, got {len(traps)}'
    trap = traps[0]
    assert trap.side_effects, 'trap must be side-effecting so DCE does not prune it'
    assert 'inc' in trap.code.as_string, f'trap must check the constraint symbol inc: {trap.code.as_string!r}'

    assert any(isinstance(n, nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive()), 'loop should be lifted to a Map'
    assert not any(
        isinstance(cf, LoopRegion) and cf.loop_variable
        for cf in sdfg.all_control_flow_regions(recursive=True)), 'no sequential loop should remain'
    assert all(e.data.wcr is None for s in sdfg.all_states() for e in s.edges()), 'the parallel Map must carry no WCR'


def test_value_preserving_under_nonzero_stride():
    # inc=1 keeps every i*inc in bounds (the realistic stride); the parallel Map
    # under the trap guard must reproduce the sequential in-place update.
    n = 64
    inc = 1
    rng = np.random.default_rng(0)
    a0, b = rng.standard_normal(n), rng.standard_normal(n)
    sdfg = affine_stride_rmw.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    got = a0.copy()
    sdfg(a=got, b=b, inc=inc, N=n)
    exp = a0 + b  # a[i*1] = a[i] + b[i] for all i
    assert np.allclose(got, exp), 'value mismatch at inc=1'


if __name__ == '__main__':
    test_symbolic_stride_emits_trap_guard_and_parallel_map()
    test_value_preserving_under_nonzero_stride()
