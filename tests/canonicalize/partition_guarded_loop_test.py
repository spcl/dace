# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Contract-based iteration-space partitioning: what it splits and what it refuses.

The kernel under test is the MPR-PPG figure's own::

    for i in range(1, N - 1):
        if i * i < N: A[i] = A[i - 1] + B[i]     # carried RAW, only under the guard
        else:         A[i] = C[i] + B[i]

Splitting it is only sound because the guard FALLS in ``i``. Every negative case below is a
predicate that looks partitionable and is not, and each asserts the SDFG comes back BIT-IDENTICAL
-- a refusal that still rewrote the graph is the failure mode that ships a miscompile quietly.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize import partition_guarded_loop as pgl
from dace.transformation.passes.canonicalize.partition_guarded_loop import PartitionGuardedLoop
from dace.transformation.passes.scalar_fission import PrivatizeScalars

N = dace.symbol('N', dtype=dace.int64)
M = dace.symbol('M', dtype=dace.int64)


@dace.program
def figure(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in range(1, N - 1):
        if i * i < N:
            A[i] = A[i - 1] + B[i]
        else:
            A[i] = C[i] + B[i]


@dace.program
def always_true(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in range(1, N - 1):
        if i < N:
            A[i] = A[i - 1] + B[i]
        else:
            A[i] = C[i] + B[i]


@dace.program
def always_false(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in range(1, N - 1):
        if i < 1:
            A[i] = A[i - 1] + B[i]
        else:
            A[i] = C[i] + B[i]


@dace.program
def non_monotone(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in range(1, N - 1):
        if (i - 3) * (i - 3) < 4:
            A[i] = A[i - 1] + B[i]
        else:
            A[i] = C[i] + B[i]


@dace.program
def undecidable(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in range(1, N - 1):
        if M * i < N:
            A[i] = A[i - 1] + B[i]
        else:
            A[i] = C[i] + B[i]


@dace.program
def cross_branch(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in range(1, N - 1):
        if i * i < N:
            A[i] = A[i - 1] + B[i]
        else:
            A[i] = A[i - 1] + C[i]


@dace.program
def data_dependent(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in range(1, N - 1):
        if C[i] < 0.0:
            A[i] = A[i - 1] + B[i]
        else:
            A[i] = C[i] + B[i]


def map_count(sdfg: dace.SDFG) -> int:
    """Number of Map scopes anywhere in ``sdfg``."""
    return sum(1 for state in sdfg.all_states() for node in state.nodes() if isinstance(node, nodes.MapEntry))


def loop_count(sdfg: dace.SDFG) -> int:
    """Number of LoopRegions anywhere in ``sdfg``."""
    return sum(1 for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion))


def parallelize(sdfg: dace.SDFG) -> int:
    """The prep + ``LoopToMap`` the canonicalization pipeline runs, so a test sees what it would.

    Without the privatization a per-iteration scalar temporary is a false write/write dependence and
    ``LoopToMap`` refuses every frontend-shaped loop, partitioned or not.
    """
    PrivatizeScalars().apply_pass(sdfg, {})
    return sdfg.apply_transformations_repeated([LoopToMap])


def reference(a: np.ndarray, b: np.ndarray, c: np.ndarray, n: int) -> None:
    """The figure's kernel, run plainly and sequentially."""
    for i in range(1, n - 1):
        if i * i < n:
            a[i] = a[i - 1] + b[i]
        else:
            a[i] = c[i] + b[i]


def inputs(n: int):
    """``(A, B, C)`` for a run of length ``n``."""
    return (np.arange(n,
                      dtype=np.float64), np.arange(n, dtype=np.float64) * 0.5, np.arange(n, dtype=np.float64) * -0.25)


def test_figure_kernel_partitions():
    """The figure's loop becomes scan + sequential prefix + one parallel Map."""
    sdfg = figure.to_sdfg(simplify=True)
    assert loop_count(sdfg) == 1 and map_count(sdfg) == 0

    assert PartitionGuardedLoop().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    # The scan that finds the cut, the sequential prefix, and the suffix still in loop form.
    assert loop_count(sdfg) == 3
    cut = next(s for s in sdfg.symbols if s.startswith(pgl.SPLIT_PREFIX))
    prefix = next(r for r in sdfg.all_control_flow_regions(recursive=True)
                  if isinstance(r, LoopRegion) and r.loop_variable and cut in r.loop_condition.as_string)
    assert str(loop_analysis.get_loop_end(prefix)) == f'{cut} - 1'

    assert parallelize(sdfg) == 1
    sdfg.validate()
    assert map_count(sdfg) == 1, 'the parallel suffix must lift into exactly one Map'
    assert loop_count(sdfg) == 2, 'the scan and the sequential prefix must survive as loops'


@pytest.mark.parametrize('n', [2, 3, 5, 9, 10, 16, 17, 25, 26, 40, 101])
def test_figure_kernel_numerics(n):
    """Partitioned output equals the plain sequential reference, at square and non-square ``N``."""
    sdfg = figure.to_sdfg(simplify=True)
    assert PartitionGuardedLoop().apply_pass(sdfg, {}) == 1
    assert parallelize(sdfg) == 1

    got, b, c = inputs(n)
    want = got.copy()
    sdfg(A=got, B=b, C=c, N=n)
    reference(want, b, c, n)
    assert np.allclose(got, want)


def test_partition_is_idempotent():
    """A second run must not cut the segments the first run made: the graph would grow forever."""
    sdfg = figure.to_sdfg(simplify=True)
    assert PartitionGuardedLoop().apply_pass(sdfg, {}) == 1
    after_first = sdfg.to_json()
    assert PartitionGuardedLoop().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == after_first


def test_always_true_predicate_stays_sequential():
    """A guard that holds on every iteration leaves nothing parallel: no partition, no Map."""
    sdfg = always_true.to_sdfg(simplify=True)
    before = sdfg.to_json()
    assert PartitionGuardedLoop().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before, 'a refused partition must leave the SDFG bit-identical'
    assert parallelize(sdfg) == 0
    assert map_count(sdfg) == 0 and loop_count(sdfg) == 1


def test_always_false_predicate_is_one_map():
    """A guard that holds on no iteration leaves the carrier dead: the whole loop is one Map."""
    sdfg = always_false.to_sdfg(simplify=True)
    before = sdfg.to_json()
    assert PartitionGuardedLoop().apply_pass(sdfg, {}) is None, 'nothing to partition: no prefix exists'
    assert sdfg.to_json() == before
    assert parallelize(sdfg) == 1
    assert map_count(sdfg) == 1 and loop_count(sdfg) == 0

    n = 40
    got, b, c = inputs(n)
    want = got.copy()
    sdfg(A=got, B=b, C=c, N=n)
    for i in range(1, n - 1):
        want[i] = c[i] + b[i]
    assert np.allclose(got, want)


@pytest.mark.parametrize('program', [non_monotone, undecidable, cross_branch, data_dependent],
                         ids=['non_monotone', 'undecidable', 'cross_branch', 'data_dependent'])
def test_refused_shapes_are_untouched(program):
    """Every predicate the analysis cannot underwrite leaves the loop bit-identical and sequential."""
    sdfg = program.to_sdfg(simplify=True)
    before = sdfg.to_json()
    assert PartitionGuardedLoop().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before, 'a refused partition must leave the SDFG bit-identical'
    assert loop_count(sdfg) == 1 and map_count(sdfg) == 0


def test_cross_branch_dependence_numerics():
    """The refused cross-branch kernel still computes what the sequential program computes."""
    sdfg = cross_branch.to_sdfg(simplify=True)
    assert PartitionGuardedLoop().apply_pass(sdfg, {}) is None
    n = 40
    got, b, c = inputs(n)
    want = got.copy()
    sdfg(A=got, B=b, C=c, N=n)
    for i in range(1, n - 1):
        want[i] = want[i - 1] + (b[i] if i * i < n else c[i])
    assert np.allclose(got, want)


@pytest.mark.parametrize('guard, falling', [
    ('i ** 2 < N', True),
    ('N > i ** 2', True),
    ('i < 4', True),
    ('i >= 4', False),
    ('N < i ** 2', False),
    ('(i - 3) ** 2 < 4', False),
    ('M * i < N', False),
])
def test_monotonicity_verdicts(guard, falling):
    """``provably_falling`` accepts only the predicates that are true on a prefix and false after.

    ``i >= 4`` and ``N < i ** 2`` RISE -- false first, true after -- so their parallel part would be
    the prefix, not the suffix, and this pass has no shape for that. ``(i - 3) ** 2 < 4`` flips
    twice, and ``M * i < N`` has a step of unknown sign. All three are undecidable-or-wrong here and
    all three must be refused, not guessed.
    """
    relation = dace.symbolic.pystr_to_symbolic(guard)
    ivar = pgl.loop_variable_in(relation, 'i')
    assert pgl.provably_falling(relation, ivar, dace.symbolic.pystr_to_symbolic('1'),
                                dace.symbolic.pystr_to_symbolic('1')) is falling


def test_always_true_gate():
    """The last iteration deciding the guard true is what makes a falling predicate useless."""
    relation = dace.symbolic.pystr_to_symbolic('i < N')
    ivar = pgl.loop_variable_in(relation, 'i')
    assert pgl.provably_true_at(relation, ivar, dace.symbolic.pystr_to_symbolic('N - 2'))
    undecided = dace.symbolic.pystr_to_symbolic('i ** 2 < N')
    assert not pgl.provably_true_at(undecided, pgl.loop_variable_in(undecided, 'i'),
                                    dace.symbolic.pystr_to_symbolic('N - 2'))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
