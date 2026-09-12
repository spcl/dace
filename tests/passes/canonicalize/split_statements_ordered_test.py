# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SplitStatements`` may separate two statements that share an array, by ORDERING the loops.

The sibling split (``rmw_confined`` + ``_split``) puts every clone in one state, so it may only
separate outputs no other group reads. That refuses the two commonest TSVC shapes outright, and both
of them distribute perfectly well once the two loops are given an order:

* ``s212`` -- ``a[i] = a[i]*c[i]; b[i] = b[i] + a[i+1]*d[i]``. The read of ``a`` is one AHEAD of the
  write, so the reader wants the original values and the reader's loop goes FIRST.
* ``s221`` -- ``a[i] = a[i] + c[i]*d[i]; b[i] = b[i-1] + a[i] + d[i]``. The read of ``a`` is at the
  SAME index the writer just wrote, so the writer's loop goes first.

Fused, either kernel is one wholly sequential loop with no Map at all. Split, both halves are
parallel work: two Maps for s212, a Map and a prefix ``Scan`` for s221.

The rules read the shared array as one element PER ITERATION, so they hold only while the writes
really are per-iteration. A value the loop carries in a scalar compares EQUAL to its own store
and would otherwise be mistaken for s221's shape; it has no legal order at all.
"""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.scan import Scan
from dace.sdfg import nodes as nd
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.split_statements import SplitStatements

N = dace.symbol('N')


@dace.program
def read_ahead(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    """TSVC s212: the second statement reads ``a`` one element ahead of the first's write."""
    for i in range(N - 1):
        a[i] = a[i] * c[i]
        b[i] = b[i] + a[i + 1] * d[i]


@dace.program
def same_index(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    """TSVC s221: the second statement reads the element the first just wrote, plus a prefix sum."""
    for i in range(1, N):
        a[i] = a[i] + c[i] * d[i]
        b[i] = b[i - 1] + a[i] + d[i]


@dace.program
def pulls_both_ways(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """Each statement reads the other's array one AHEAD, so each demands to go first.

    ``a`` is written here and read ahead by the ``b`` statement, which asks for b-then-a; ``b`` is
    written here and read ahead by the ``a`` statement, which asks for a-then-b. Neither order
    reproduces the fused loop, so no split is legal and the pass must leave the SDFG alone.
    """
    for i in range(1, N - 1):
        a[i] = a[i] * c[i] + b[i + 1]
        b[i] = b[i] * c[i] + a[i + 1]


@dace.program
def carried_scalar(a: dace.float64[N], b: dace.float64[N]):
    """The shared value is a SCALAR the loop CARRIES, not one element per iteration.

    Every direction rule reads the shared datum as one element per iteration, and ``s`` is the same
    element in all of them: it compares EQUAL to its own store, which the same-index rule would
    otherwise read as s221's shape. Writer first gives the total in every ``b[i]``, reader first
    gives the initial value, and the fused loop gave the running sum -- so no order is legal.
    """
    s = 0.0
    for i in range(N):
        s = s + a[i]
        b[i] = s


def _loops(sdfg: dace.SDFG) -> list:
    """The top-level loops, in execution order."""
    order, seen, queue = [], set(), [sdfg.start_block]
    while queue:
        block = queue.pop(0)
        if id(block) in seen:
            continue
        seen.add(id(block))
        order.append(block)
        queue.extend(e.dst for e in sdfg.out_edges(block))
    return [b for b in order if isinstance(b, LoopRegion)]


def _writes(loop: LoopRegion) -> set:
    """Array names ``loop`` stores to."""
    return {
        n.data
        for st in loop.all_states()
        for n in st.data_nodes() if any(e.data is not None and not e.data.is_empty() for e in st.in_edges(n))
    }


def _run(program, sdfg: dace.SDFG, n: int = 64, seed: int = 5) -> None:
    """Assert ``sdfg`` still computes what the unmodified program computes."""
    rng = np.random.default_rng(seed)
    args = {name: rng.random(n) for name in 'abcd' if name in program.f.__code__.co_varnames}
    ref = {k: v.copy() for k, v in args.items()}
    _reference(program.name, ref, n)
    got = {k: v.copy() for k, v in args.items()}
    sdfg(N=n, **got)
    for name, value in got.items():
        assert np.allclose(value, ref[name], equal_nan=True), f'{name} diverges from the oracle'


def _reference(which: str, arr: dict, n: int) -> None:
    """Plain-numpy oracle, written out iteration by iteration."""
    a, b, c = arr['a'], arr['b'], arr.get('c')
    d = arr.get('d')
    if which.endswith('read_ahead'):
        for i in range(n - 1):
            a[i] = a[i] * c[i]
            b[i] = b[i] + a[i + 1] * d[i]
    elif which.endswith('same_index'):
        for i in range(1, n):
            a[i] = a[i] + c[i] * d[i]
            b[i] = b[i - 1] + a[i] + d[i]
    else:
        for i in range(1, n - 1):
            a[i] = a[i] * c[i] + b[i + 1]
            b[i] = b[i] * c[i] + a[i + 1]


def test_read_ahead_puts_the_reader_first():
    """s212: the loop that READS ``a`` must run before the loop that overwrites it."""
    sdfg = read_ahead.to_sdfg(simplify=True)
    assert SplitStatements().apply_pass(sdfg, {}) == 1, 'the ordered split must fire'
    sdfg.validate()

    loops = _loops(sdfg)
    assert len(loops) == 2, f'one loop per statement: {[l.label for l in loops]}'
    assert 'b' in _writes(loops[0]) and 'a' not in _writes(loops[0]), 'the reader of a runs first'
    assert 'a' in _writes(loops[1]), 'the writer of a runs second'
    _run(read_ahead, sdfg)


def test_same_index_read_puts_the_writer_first():
    """s221: the loop that WRITES ``a`` must run before the loop that reads the same element."""
    sdfg = same_index.to_sdfg(simplify=True)
    assert SplitStatements().apply_pass(sdfg, {}) == 1, 'the ordered split must fire'
    sdfg.validate()

    loops = _loops(sdfg)
    assert len(loops) == 2, f'one loop per statement: {[l.label for l in loops]}'
    assert 'a' in _writes(loops[0]) and 'b' not in _writes(loops[0]), 'the writer of a runs first'
    assert 'b' in _writes(loops[1]), 'the reader of a runs second'
    _run(same_index, sdfg)


def test_contradictory_directions_are_refused():
    """No order reproduces the fused loop, so the pass must leave the SDFG bit-identical."""
    sdfg = pulls_both_ways.to_sdfg(simplify=True)
    before = sdfg.hash_sdfg()
    assert SplitStatements().apply_pass(sdfg, {}) is None, 'a contradictory pair must not split'
    assert sdfg.hash_sdfg() == before, 'a pass that does not apply must not mutate'


@pytest.mark.parametrize('program,maps,scans', [(read_ahead, 2, 0), (same_index, 1, 1)])
def test_canonicalize_parallelizes_the_split_pair(program, maps, scans):
    """End to end: the fused loop has no Map at all, and after the split neither half is a loop."""
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)
    sdfg.validate()

    assert not _loops(sdfg), 'no sequential loop may survive'
    assert sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry)) == maps
    assert sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan)) == scans
    _run(program, sdfg)


def test_carried_scalar_is_refused():
    """The running sum is carried in a scalar, so neither order reproduces the fused loop."""
    sdfg = carried_scalar.to_sdfg(simplify=True)
    before = sdfg.hash_sdfg()
    assert SplitStatements().apply_pass(sdfg, {}) is None, 'a carried scalar must not split'
    assert sdfg.hash_sdfg() == before, 'a pass that does not apply must not mutate'


def test_carried_scalar_keeps_the_running_sum():
    """End to end: canonicalize must still compute the prefix sum, not the total."""
    n = 64
    a = np.random.default_rng(7).random(n)
    sdfg = carried_scalar.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)
    sdfg.validate()
    got = np.zeros(n)
    sdfg(a=a.copy(), b=got, N=n)
    assert np.allclose(got, np.cumsum(a)), 'a split loop would read the final total in every b[i]'


if __name__ == '__main__':
    test_read_ahead_puts_the_reader_first()
    test_same_index_read_puts_the_writer_first()
    test_contradictory_directions_are_refused()
    test_carried_scalar_is_refused()
    test_carried_scalar_keeps_the_running_sum()
    test_canonicalize_parallelizes_the_split_pair(read_ahead, 2, 0)
    test_canonicalize_parallelizes_the_split_pair(same_index, 1, 1)
