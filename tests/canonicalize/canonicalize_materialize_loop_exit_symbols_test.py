# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`MaterializeLoopExitSymbols`.

A loop-defined symbol (``k = k + step`` on a body interstate edge) whose
final value is read after the loop blocks ``LoopToMap``. The pass materialises
the closed-form exit value under a fresh unique name and rewrites every
post-loop reader to use it, so the original symbol is no longer "used after the
loop" and the body can parallelise.
"""
import numpy as np
import pytest

import dace
from dace import symbolic
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.materialize_loop_exit_symbols import (MaterializeLoopExitSymbols,
                                                                                   POST_PREFIX)

N = dace.symbol('N')
step = dace.symbol('step')


def _has_loop_exit_sym(sdfg, base_name):
    return any(s.startswith(f"{POST_PREFIX}{base_name}_") for s in sdfg.symbols)


def test_post_loop_iv_symbol_materialised_with_unique_name():
    """Build an SDFG by hand: pre-loop ``k=0`` + loop body ``k = k + step`` +
    post-loop tasklet reading ``k``. The pass should add a ``_loop_exit_k_<N>``
    symbol whose post-loop assignment is the closed form, and rewrite the
    post-loop tasklet to read it."""
    sdfg = dace.SDFG('mat_iv_post')
    sdfg.add_symbol('k', dace.int64)
    sdfg.add_symbol('step', dace.int64)
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('out', [1], dace.int64)

    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'k': '0'}))
    body = loop.add_state('body', is_start_block=True)
    body2 = loop.add_state('body2')
    loop.add_edge(body, body2, dace.InterstateEdge(assignments={'k': 'k + step'}))

    post = sdfg.add_state('post')
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    out_w = post.add_write('out')
    t = post.add_tasklet('write_k', {}, {'__o'}, '__o = k', language=dace.dtypes.Language.Python)
    post.add_edge(t, '__o', out_w, None, dace.Memlet(data='out', subset='0'))

    res = MaterializeLoopExitSymbols().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _has_loop_exit_sym(sdfg, 'k')
    assert '_loop_exit_k' in t.code.as_string, (
        f"post-loop tasklet should read the materialised symbol; got code={t.code.as_string!r}")


def test_no_post_loop_use_is_noop():
    """If the loop-defined symbol is never read after the loop, the pass refuses."""
    sdfg = dace.SDFG('mat_iv_unused')
    sdfg.add_symbol('k', dace.int64)
    sdfg.add_symbol('step', dace.int64)
    sdfg.add_symbol('N', dace.int64)

    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'k': '0'}))
    body = loop.add_state('body', is_start_block=True)
    body2 = loop.add_state('body2')
    loop.add_edge(body, body2, dace.InterstateEdge(assignments={'k': 'k + step'}))
    post = sdfg.add_state('post')
    sdfg.add_edge(loop, post, dace.InterstateEdge())  # post does not read k

    res = MaterializeLoopExitSymbols().apply_pass(sdfg, {})
    assert res is None
    assert not _has_loop_exit_sym(sdfg, 'k')


def test_a_step_over_an_enclosing_loop_iterator_is_materialised():
    """``for j: k = 0; for i < N: k = k + j; out[j] = k``: ``j`` is invariant in the inner loop.

    Neither iterator is in ``sdfg.symbols``; a step test that only asks that table calls ``j`` unknown
    and keeps the inner loop pinned by its post-loop reader.
    """
    sdfg = dace.SDFG('mat_iv_enclosing_step')
    sdfg.add_symbol('k', dace.int64)
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('out', [3], dace.int64)
    outer = LoopRegion('outer', 'j < 3', 'j', 'j = 0', 'j = j + 1')
    sdfg.add_node(outer, is_start_block=True)
    seed = outer.add_state('seed', is_start_block=True)
    inner = LoopRegion('inner', 'i < N', 'i', 'i = 0', 'i = i + 1')
    outer.add_node(inner)
    outer.add_edge(seed, inner, dace.InterstateEdge(assignments={'k': '0'}))
    body = inner.add_state('body', is_start_block=True)
    body2 = inner.add_state('body2')
    inner.add_edge(body, body2, dace.InterstateEdge(assignments={'k': 'k + j'}))
    post = outer.add_state('post')
    outer.add_edge(inner, post, dace.InterstateEdge())
    tasklet = post.add_tasklet('write_k', {}, {'__o'}, '__o = k')
    post.add_edge(tasklet, '__o', post.add_write('out'), None, dace.Memlet('out[j]'))

    assert MaterializeLoopExitSymbols().apply_pass(sdfg, {}) == 1

    sdfg.validate()
    assert _has_loop_exit_sym(sdfg, 'k')
    assert 'j' not in sdfg.symbols


def int32_iterator_read_after_its_loop() -> dace.SDFG:
    """``for i = M32; i <= N32; i += S32`` then ``out[0] = a[i - S32]``, the last element visited.

    Symbolic bounds and stride with no integer literal, so the loop scope types ``i`` as int32;
    ``i`` itself is registered in no symbol table, since the loop binds it.
    """
    sdfg = dace.SDFG('mat_iter_post_int32')
    for name in ('M32', 'N32', 'S32'):
        sdfg.add_symbol(name, dace.int32)
    sdfg.add_array('a', ['N32 + 1'], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('loop', 'i <= N32', 'i', 'i = M32', 'i = i + S32')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    loop.add_state('body', is_start_block=True)
    post = sdfg.add_state('post')
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    tasklet = post.add_tasklet('read_last', {'__a'}, {'__o'}, '__o = __a')
    post.add_edge(post.add_read('a'), None, tasklet, '__a', dace.Memlet('a[i - S32]'))
    post.add_edge(tasklet, '__o', post.add_write('out'), None, dace.Memlet('out[0]'))
    return sdfg


def post_loop_read_subset(sdfg: dace.SDFG) -> symbolic.SymbolicType:
    (edge, ) = [e for s in sdfg.states() if s.label == 'post' for e in s.edges() if e.data.data == 'a']
    return edge.data.subset[0][0]


def test_an_int32_iterator_exit_value_is_declared_at_the_width_its_loop_gives_it():
    sdfg = int32_iterator_read_after_its_loop()

    assert MaterializeLoopExitSymbols().apply_pass(sdfg, {}) == 1

    sdfg.validate()
    (exit_name, ) = [s for s in sdfg.symbols if s.startswith(f"{POST_PREFIX}i_")]
    assert sdfg.symbols[exit_name] == dace.int32
    assert 'i' not in sdfg.symbols
    # Declared at any other width, the reader's re-parsed exit symbol is a second symbol of the same
    # name and the difference below does not cancel.
    declared = symbolic.symbol(exit_name, sdfg.symbols[exit_name]) - symbolic.symbol('S32', dace.int32)
    assert symbolic.simplify(post_loop_read_subset(sdfg) - declared) == 0


def test_an_int32_iterator_exit_value_reads_the_last_visited_element():
    sdfg = int32_iterator_read_after_its_loop()
    a = np.arange(11, dtype=np.float64) * 1.5
    out = np.zeros(1, dtype=np.float64)
    assert MaterializeLoopExitSymbols().apply_pass(sdfg, {}) == 1

    sdfg(a=a, out=out, M32=1, N32=10, S32=3)

    assert out[0] == a[10]  # i visits 1, 4, 7, 10 and exits at 13


def two_loops_sharing_an_unregistered_iterator() -> tuple[dace.SDFG, LoopRegion]:
    """``for i < N - 1: a[i] = 1; for i < N - 1: d[i] = a[i + 1]``, the fission shape of TSVC ``s1244``.

    The second loop binds ``i`` afresh, and neither loop registers it in ``sdfg.symbols``.
    """
    sdfg = dace.SDFG('mat_shared_iterator')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('a', ['N'], dace.float64)
    sdfg.add_array('d', ['N'], dace.float64)
    first = LoopRegion('first', 'i < N - 1', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(first, is_start_block=True)
    store = first.add_state('store', is_start_block=True)
    one = store.add_tasklet('one', {}, {'__o'}, '__o = 1.0')
    store.add_edge(one, '__o', store.add_write('a'), None, dace.Memlet('a[i]'))
    second = LoopRegion('second', 'i < N - 1', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(second)
    sdfg.add_edge(first, second, dace.InterstateEdge())
    read = second.add_state('read_ahead', is_start_block=True)
    copy = read.add_tasklet('copy', {'__x'}, {'__o'}, '__o = __x')
    read.add_edge(read.add_read('a'), None, copy, '__x', dace.Memlet('a[i + 1]'))
    read.add_edge(copy, '__o', read.add_write('d'), None, dace.Memlet('d[i]'))
    return sdfg, second


def test_a_later_loop_rebinding_the_iterator_is_not_a_read_of_its_exit_value():
    """Rewriting the second loop's own ``i`` to the first loop's exit value made it read ``a[N]`` (s1244)."""
    sdfg, second = two_loops_sharing_an_unregistered_iterator()

    assert MaterializeLoopExitSymbols().apply_pass(sdfg, {}) is None

    sdfg.validate()
    assert not _has_loop_exit_sym(sdfg, 'i')
    (edge, ) = [e for state in second.all_states() for e in state.edges() if e.data.data == 'a']
    assert str(edge.data.subset) == 'i + 1'


@dace.program
def count_trips(out: dace.float64[1]):
    na = 0
    for _ in range(6):
        na += 1
    out[0] = na


@dace.program
def count_trips_until_negative(a: dace.float64[N], out: dace.float64[1]):
    na = 0
    for _ in range(6):
        na += 1
        if a[na - 1] < 0.0:
            break
    out[0] = na


def test_a_counter_exit_value_counts_each_trip_once():
    """The exit value is ``seed + step * trips`` with the PRE-loop seed; read after the loop, the counter
    already holds its exit value, and adding the trips to it again counted every trip twice (ls3df_scf)."""
    sdfg = count_trips.to_sdfg(simplify=True)
    assert MaterializeLoopExitSymbols().apply_pass(sdfg, {}) == 1
    out = np.zeros(1)

    sdfg(out=out)

    assert out[0] == 6, out


def test_a_loop_left_by_break_keeps_its_counted_exit_value():
    """A break makes the trip count data-dependent, so the closed form over the full range is not the exit
    value; materialising it grew a post-loop extent past its allocation (ls3df_scf's Lanczos ``na``)."""
    sdfg = count_trips_until_negative.to_sdfg(simplify=True)
    assert MaterializeLoopExitSymbols().apply_pass(sdfg, {}) is None
    out = np.zeros(1)

    sdfg(a=np.array([1.0, 2.0, -3.0, 4.0, 5.0, 6.0]), out=out, N=6)

    assert out[0] == 3, out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
