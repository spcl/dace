# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SplitStatements`` distributing a STRAIGHT-LINE loop, one loop per output.

``canonicalize_straight_line_split_test.py`` pins the route through a ``NestedSDFG`` body. This
file pins the other one: at the 'prep' stage a plain loop body is a FLAT ``SDFGState`` with no
``NestedSDFG`` anywhere, so nothing in the pass would ever see it. The loop path outlines the whole
``LoopRegion``, clones it per independent output group and inlines every clone back, so::

    for i in range(1, N):
        s1 = x[i]
        a[i] = a[i - 1] + s1   # sequential carry on a
        b[i] = y[i] * 2.0      # independent, parallel

comes out as TWO loops -- the carry loop and the parallel one -- from ``SplitStatements`` alone,
with no ``LoopFission`` involved.

The refusals are checked on the SDFG HASH, not just on the return value: the outlining is not free
to undo, so every refusal has to be decided before the loop is touched.
"""
import copy
import os
import subprocess
import sys

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.canonicalize.split_statements import SplitStatements

M = dace.symbol('M')
N = dace.symbol('N')


def _loop_sdfg(name, arrays=('a', 'b', 'x', 'y')):
    """``for i in range(1, N)`` over ``arrays``, with an empty FLAT body state."""
    sdfg = dace.SDFG(name)
    for nm in arrays:
        sdfg.add_array(nm, [N], dace.float64)
    loop = LoopRegion('loop', 'i < N', 'i', 'i = 1', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    return sdfg, loop, loop.add_state('body', is_start_block=True)


def carry_loop(name, shared_temp=False, ordering_edge=False):
    """The motivating kernel, flat: ``a[i] = a[i-1] + s1`` and ``b[i] = y[i] * 2``.

    :param name: SDFG name.
    :param shared_temp: Feed the temp to BOTH statements instead of only the carry.
    :param ordering_edge: Also add the empty (ordering-only) memlet the frontend leaves between the
        two chains when they reuse one temporary name.
    """
    sdfg, _loop, st = _loop_sdfg(name)
    sdfg.add_scalar('s1', dace.float64, transient=True)

    t0 = st.add_tasklet('t0', {'inp'}, {'out'}, 'out = inp')
    ws1 = st.add_access('s1')
    st.add_edge(st.add_read('x'), None, t0, 'inp', dace.Memlet('x[i]'))
    st.add_edge(t0, 'out', ws1, None, dace.Memlet('s1'))

    tadd = st.add_tasklet('_Add_', {'prev', 'sv'}, {'out'}, 'out = prev + sv')
    st.add_edge(st.add_read('a'), None, tadd, 'prev', dace.Memlet('a[i - 1]'))
    st.add_edge(ws1, None, tadd, 'sv', dace.Memlet('s1'))
    st.add_edge(tadd, 'out', st.add_write('a'), None, dace.Memlet('a[i]'))

    ry = st.add_read('y')
    if shared_temp:
        tmul = st.add_tasklet('_Mult_', {'yy', 'sv'}, {'out'}, 'out = yy * sv')
        st.add_edge(ws1, None, tmul, 'sv', dace.Memlet('s1'))
    else:
        tmul = st.add_tasklet('_Mult_', {'yy'}, {'out'}, 'out = yy * 2.0')
    st.add_edge(ry, None, tmul, 'yy', dace.Memlet('y[i]'))
    st.add_edge(tmul, 'out', st.add_write('b'), None, dace.Memlet('b[i]'))
    if ordering_edge:
        st.add_nedge(tadd, ry, dace.Memlet())
    return sdfg


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]


def _written(sdfg, loop):
    """The distinct GLOBAL array names a loop body stores to (its private temps are not outputs)."""
    stored = dict.fromkeys(n.data for s in loop.all_states() for n in s.data_nodes() if s.in_degree(n) > 0)
    return sorted(nm for nm in stored if not sdfg.arrays[nm].transient)


def _split(sdfg):
    return SplitStatements().apply_pass(sdfg, {})


def _run(sdfg, n=32):
    """Compile and run over fixed inputs; returns the two written arrays."""
    rng = np.random.default_rng(7)
    args = {'a': rng.random(n), 'b': np.zeros(n), 'x': rng.random(n), 'y': rng.random(n)}
    sdfg.compile()(**args, N=n)
    return args['a'], args['b']


def run_arrays(sdfg, n=32, seed=7):
    """Compile and run over deterministic inputs; returns every non-transient array after the run.

    Names are sorted so a split SDFG and its unsplit deepcopy draw the SAME inputs even though the
    split added transients of its own -- the point is to compare the two runs, not to seed prettily.
    """
    rng = np.random.default_rng(seed)
    args = {nm: rng.random(n) for nm in sorted(nm for nm, desc in sdfg.arrays.items() if not desc.transient)}
    sdfg.compile()(**args, N=n)
    return args


def assert_matches_fused(split, fused):
    """The distributed loops must compute what the one fused loop computed, bit for bit."""
    want, got = run_arrays(fused), run_arrays(split)
    for name, expected in want.items():
        assert np.array_equal(got[name], expected), f'{name} diverges from the fused loop'


def _refuses(sdfg):
    """The pass leaves ``sdfg`` byte-identical."""
    before = sdfg.hash_sdfg()
    fired = _split(sdfg)
    return fired is None and sdfg.hash_sdfg() == before


# ===========================================================================
# The flat loop is distributed into one loop per statement.
# ===========================================================================
def test_flat_loop_splits_into_two_loops():
    """One loop in, two loops out -- the carry and the parallel statement, each self-contained."""
    sdfg = carry_loop('flat_split')
    assert len(_loops(sdfg)) == 1
    assert not [n for l in _loops(sdfg) for s in l.all_states() for n in s.nodes() if isinstance(n, nodes.NestedSDFG)]

    assert _split(sdfg) == 1
    loops = _loops(sdfg)
    assert len(loops) == 2
    assert sorted(_written(sdfg, l) for l in loops) == [['a'], ['b']]
    sdfg.validate()


def test_flat_loop_split_preserves_values():
    """The real gate: two loops compute exactly what the one loop computed."""
    ref_a, ref_b = _run(carry_loop('flat_ref'))
    sdfg = carry_loop('flat_cand')
    assert _split(sdfg) == 1
    got_a, got_b = _run(sdfg)
    assert np.array_equal(got_a, ref_a)
    assert np.array_equal(got_b, ref_b)


def test_ordering_edge_does_not_merge_the_loop_groups():
    """An empty memlet constrains execution ORDER, not dataflow, so it must not bind the outputs.

    It rides along verbatim inside whichever clone holds both its endpoints.
    """
    sdfg = carry_loop('flat_order', ordering_edge=True)
    assert _split(sdfg) == 1
    assert len(_loops(sdfg)) == 2
    sdfg.validate()


def test_ordering_edge_loop_split_preserves_values():
    ref_a, ref_b = _run(carry_loop('flat_order_ref', ordering_edge=True))
    sdfg = carry_loop('flat_order_cand', ordering_edge=True)
    assert _split(sdfg) == 1
    got_a, got_b = _run(sdfg)
    assert np.array_equal(got_a, ref_a)
    assert np.array_equal(got_b, ref_b)


def test_shared_temp_is_recomputed_in_each_loop():
    """``s1 = x[i]`` feeding BOTH statements is a private temp of each clone, not a hand-off.

    ``nest_sdfg_subgraph`` hands a loop-local transient back as a CONNECTOR (its locality test
    already recurses into the region being outlined), and left there both loops would write the one
    outer ``s1``. It is moved inside instead, so each loop derives its own from ``x``.
    """
    sdfg = carry_loop('flat_shared', shared_temp=True)
    assert _split(sdfg) == 1
    loops = _loops(sdfg)
    assert len(loops) == 2
    # Two loops, two DISTINCT private temps -- neither hands a value to the other.
    temps = [
        sorted(dict.fromkeys(n.data for s in l.all_states() for n in s.data_nodes() if sdfg.arrays[n.data].transient))
        for l in loops
    ]
    assert all(t for t in temps), temps
    assert not set(temps[0]) & set(temps[1]), temps


def test_shared_temp_loop_split_preserves_values():
    ref_a, ref_b = _run(carry_loop('flat_shared_ref', shared_temp=True))
    sdfg = carry_loop('flat_shared_cand', shared_temp=True)
    assert _split(sdfg) == 1
    got_a, got_b = _run(sdfg)
    assert np.array_equal(got_a, ref_a)
    assert np.array_equal(got_b, ref_b)


def test_three_parallel_outputs_stay_in_one_loop():
    """Nothing carries, so the loop is already parallel -- splitting it would only add passes.

    The split exists to free parallel work from a sequential recurrence, not to give every
    statement a loop: three full-length sweeps where one does is pure cost.
    """
    sdfg, _loop, st = _loop_sdfg('flat_three', arrays=('a', 'b', 'c', 'x'))
    for out in ('a', 'b', 'c'):
        t = st.add_tasklet('t_' + out, {'xx'}, {'out'}, 'out = xx * 2.0')
        st.add_edge(st.add_read('x'), None, t, 'xx', dace.Memlet('x[i]'))
        st.add_edge(t, 'out', st.add_write(out), None, dace.Memlet(f'{out}[i]'))
    assert _refuses(sdfg)


def _carry(st, out, src='x'):
    t = st.add_tasklet('add_' + out, {'prev', 'xx'}, {'o'}, 'o = prev + xx')
    st.add_edge(st.add_read(out), None, t, 'prev', dace.Memlet(f'{out}[i - 1]'))
    st.add_edge(st.add_read(src), None, t, 'xx', dace.Memlet(f'{src}[i]'))
    st.add_edge(t, 'o', st.add_write(out), None, dace.Memlet(f'{out}[i]'))


def _parallel(st, out, src='x'):
    t = st.add_tasklet('mul_' + out, {'xx'}, {'o'}, 'o = xx * 2.0')
    st.add_edge(st.add_read(src), None, t, 'xx', dace.Memlet(f'{src}[i]'))
    st.add_edge(t, 'o', st.add_write(out), None, dace.Memlet(f'{out}[i]'))


def test_carry_plus_two_parallel_yields_exactly_two_loops():
    """ONE loop for the recurrence and ONE for all the parallel work -- not one loop per output."""
    sdfg, _loop, st = _loop_sdfg('flat_carry_two_par', arrays=('a', 'b', 'c', 'x'))
    _carry(st, 'a')
    _parallel(st, 'b')
    _parallel(st, 'c')
    assert _split(sdfg) == 1
    assert sorted(_written(sdfg, l) for l in _loops(sdfg)) == [['a'], ['b', 'c']]
    sdfg.validate()


def test_two_recurrences_share_one_loop_and_the_parallel_work_is_peeled():
    """Two carries stay TOGETHER: they are both sequential, so separating them buys no parallelism
    and costs a second full-length sweep. Only the parallel statement is peeled off."""
    sdfg, _loop, st = _loop_sdfg('flat_two_carries', arrays=('a', 'b', 'c', 'x'))
    _carry(st, 'a')
    _carry(st, 'b')
    _parallel(st, 'c')
    assert _split(sdfg) == 1
    assert sorted(_written(sdfg, l) for l in _loops(sdfg)) == [['a', 'b'], ['c']]
    sdfg.validate()


def test_only_recurrences_leaves_the_loop_alone():
    """A reduction hand-unrolled into several accumulators has NOTHING parallel to peel.

    Splitting it per accumulator would be one full sweep each, gain no parallelism, and stop the
    re-roll that lifts the whole thing to a single ``Reduce`` from matching. Measured on
    ``canonicalize_reroll_unrolled_test.py::test_unroll_reduction_11_accs_value_and_reduce``, where
    an eleven-way split also brought back the half-sum the re-roll exists to avoid.
    """
    sdfg, _loop, st = _loop_sdfg('flat_only_carries', arrays=('a', 'b', 'x'))
    _carry(st, 'a')
    _carry(st, 'b')
    assert _refuses(sdfg)


def nested_carry(name):
    """An outer ``j`` loop wrapping the straight-line ``i`` loop -- the split target is inner."""
    sdfg = dace.SDFG(name)
    for nm in ('a', 'b', 'x', 'y'):
        sdfg.add_array(nm, [M, N], dace.float64)
    outer = LoopRegion('outer', 'j < M', 'j', 'j = 0', 'j = j + 1')
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion('inner', 'i < N', 'i', 'i = 1', 'i = i + 1')
    outer.add_node(inner, is_start_block=True)
    st = inner.add_state('body', is_start_block=True)
    tadd = st.add_tasklet('_Add_', {'prev', 'xx'}, {'out'}, 'out = prev + xx')
    st.add_edge(st.add_read('a'), None, tadd, 'prev', dace.Memlet('a[j, i - 1]'))
    st.add_edge(st.add_read('x'), None, tadd, 'xx', dace.Memlet('x[j, i]'))
    st.add_edge(tadd, 'out', st.add_write('a'), None, dace.Memlet('a[j, i]'))
    tmul = st.add_tasklet('_Mult_', {'yy'}, {'out'}, 'out = yy * 2.0')
    st.add_edge(st.add_read('y'), None, tmul, 'yy', dace.Memlet('y[j, i]'))
    st.add_edge(tmul, 'out', st.add_write('b'), None, dace.Memlet('b[j, i]'))
    return sdfg


def _run_2d(sdfg, m=8, n=16):
    rng = np.random.default_rng(3)
    args = {'a': rng.random((m, n)), 'b': np.zeros((m, n)), 'x': rng.random((m, n)), 'y': rng.random((m, n))}
    sdfg.compile()(**args, M=m, N=n)
    return args['a'], args['b']


def test_inner_loop_of_a_nest_splits_and_preserves_values():
    """The outer loop is refused (its body is not a single state); the inner one distributes."""
    ref_a, ref_b = _run_2d(nested_carry('nest_ref'))
    sdfg = nested_carry('nest_cand')
    assert _split(sdfg) == 1
    assert len(_loops(sdfg)) == 3  # the outer loop plus the two it now contains
    sdfg.validate()
    got_a, got_b = _run_2d(sdfg)
    assert np.array_equal(got_a, ref_a)
    assert np.array_equal(got_b, ref_b)


def test_loop_split_is_idempotent():
    """Every loop the split leaves behind writes ONE output, so a second run has nothing to do."""
    sdfg = carry_loop('flat_idem')
    assert _split(sdfg) == 1
    before = sdfg.hash_sdfg()
    assert _split(sdfg) is None
    assert sdfg.hash_sdfg() == before
    sdfg.validate()


_DETERMINISM = """
import runpy, sys
mod = runpy.run_path(sys.argv[1])
sdfg = mod['carry_loop']('det')
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
SplitStatements().apply_pass(sdfg, {})
print(sdfg.hash_sdfg())
"""


def test_loop_split_is_deterministic():
    """Same input, same SDFG -- under two different string-hash seeds.

    The split names states and connectors out of the clones it makes, so any set iteration on the
    way there would show up here as a different hash.
    """
    hashes = []
    for seed in ('0', '12345'):
        env = {**os.environ, 'PYTHONHASHSEED': seed}
        out = subprocess.run([sys.executable, '-c', _DETERMINISM, __file__],
                             capture_output=True,
                             text=True,
                             check=True,
                             env=env)
        hashes.append(out.stdout.strip().splitlines()[-1])
    assert hashes[0] == hashes[1]


# ===========================================================================
# Refusals -- each one must leave the loop byte-identical.
# ===========================================================================
def test_rmw_read_by_another_group_is_ordered_writer_first():
    """``a[i] = a[i-1] + x[i]; b[i] = a[i-1] * 2`` -- ``b`` reads ``a`` one BEHIND the write.

    The sibling split refuses this: it puts every clone in one state, so it may only separate outputs
    no other group reads. The ORDERED split does not have to -- ``a[i-1]`` is written at iteration
    ``i-1`` and never touched again, so the value ``b`` wants is already final by the time the
    carry loop ends. Writer first, and the numbers say so.
    """
    sdfg, _loop, st = _loop_sdfg('flat_cross_rmw')
    ra = st.add_read('a')
    tadd = st.add_tasklet('_Add_', {'prev', 'xx'}, {'out'}, 'out = prev + xx')
    st.add_edge(ra, None, tadd, 'prev', dace.Memlet('a[i - 1]'))
    st.add_edge(st.add_read('x'), None, tadd, 'xx', dace.Memlet('x[i]'))
    st.add_edge(tadd, 'out', st.add_write('a'), None, dace.Memlet('a[i]'))
    tmul = st.add_tasklet('_Mult_', {'av'}, {'out'}, 'out = av * 2.0')
    st.add_edge(ra, None, tmul, 'av', dace.Memlet('a[i - 1]'))
    st.add_edge(tmul, 'out', st.add_write('b'), None, dace.Memlet('b[i]'))

    fused = copy.deepcopy(sdfg)
    assert _split(sdfg) == 1, 'the ordered split must fire'
    loops = _loops(sdfg)
    assert len(loops) == 2
    assert [_written(sdfg, l) for l in loops] == [['a'], ['b']], 'the writer of a must run first'
    sdfg.validate()
    assert_matches_fused(sdfg, fused)


def test_write_read_across_groups_is_ordered_writer_first():
    """``c[i] = x[i]; b[i] = c[i-1]`` -- ``b`` sees the finished ``c``, and that is the same ``c``.

    ``c[i-1]`` is stored once, at iteration ``i-1``, so "the finished array" and "what the fused
    loop had written by then" hold the same value at every index the reader touches. Only the
    sibling split, which cannot order anything, has to refuse this.
    """
    sdfg, _loop, st = _loop_sdfg('flat_war', arrays=('b', 'c', 'x'))
    t0 = st.add_tasklet('t0', {'xx'}, {'out'}, 'out = xx')
    st.add_edge(st.add_read('x'), None, t0, 'xx', dace.Memlet('x[i]'))
    st.add_edge(t0, 'out', st.add_write('c'), None, dace.Memlet('c[i]'))
    t1 = st.add_tasklet('t1', {'cc'}, {'out'}, 'out = cc')
    st.add_edge(st.add_read('c'), None, t1, 'cc', dace.Memlet('c[i - 1]'))
    st.add_edge(t1, 'out', st.add_write('b'), None, dace.Memlet('b[i]'))

    fused = copy.deepcopy(sdfg)
    assert _split(sdfg) == 1, 'the ordered split must fire'
    loops = _loops(sdfg)
    assert len(loops) == 2
    assert [_written(sdfg, l) for l in loops] == [['c'], ['b']], 'the writer of c must run first'
    sdfg.validate()
    assert_matches_fused(sdfg, fused)


def test_three_groups_are_ordered_by_topological_sort():
    """Three outputs, two ordering constraints, one legal sequence.

    ``a[i-1] = b[i] + x[i]`` / ``d[i] = a[i] * y[i]`` / ``b[i] = z[i] * w[i]``. ``d`` reads ``a``
    one AHEAD of the write, so the reader goes first; ``a`` reads ``b`` at the index ``b``'s own
    group writes, off an access node with no producer, so the reader goes first there too. That
    leaves ``d``, then ``a``, then ``b`` -- an order two groups cannot express, which is the whole
    point of sorting the constraints instead of flipping a bit.
    """
    sdfg, _loop, st = _loop_sdfg('flat_three_groups', arrays=('a', 'b', 'd', 'x', 'y', 'z', 'w'))
    t_a = st.add_tasklet('_Add_', {'bb', 'xx'}, {'out'}, 'out = bb + xx')
    st.add_edge(st.add_read('b'), None, t_a, 'bb', dace.Memlet('b[i]'))
    st.add_edge(st.add_read('x'), None, t_a, 'xx', dace.Memlet('x[i]'))
    st.add_edge(t_a, 'out', st.add_write('a'), None, dace.Memlet('a[i - 1]'))
    t_d = st.add_tasklet('_Mult_', {'aa', 'yy'}, {'out'}, 'out = aa * yy')
    st.add_edge(st.add_read('a'), None, t_d, 'aa', dace.Memlet('a[i]'))
    st.add_edge(st.add_read('y'), None, t_d, 'yy', dace.Memlet('y[i]'))
    st.add_edge(t_d, 'out', st.add_write('d'), None, dace.Memlet('d[i]'))
    t_b = st.add_tasklet('_Mult_b', {'zz', 'ww'}, {'out'}, 'out = zz * ww')
    st.add_edge(st.add_read('z'), None, t_b, 'zz', dace.Memlet('z[i]'))
    st.add_edge(st.add_read('w'), None, t_b, 'ww', dace.Memlet('w[i]'))
    st.add_edge(t_b, 'out', st.add_write('b'), None, dace.Memlet('b[i]'))

    fused = copy.deepcopy(sdfg)
    assert _split(sdfg) == 1, 'the ordered split must fire'
    loops = _loops(sdfg)
    assert len(loops) == 3, 'one loop per group, not the two the pairwise rule could emit'
    assert [_written(sdfg, l) for l in loops] == [['d'], ['a'], ['b']]
    sdfg.validate()
    assert_matches_fused(sdfg, fused)


def test_elementwise_rmw_is_not_a_recurrence():
    """``a[i] = a[i] + x[i]`` beside ``e[i] = e[i-1] * y[i]`` -- only ``e`` carries.

    Both arrays are read AND written by the loop, which is what a name-level test sees. Only ``e``
    reads an element other than the one it writes, so only ``e`` forces its loop to stay
    sequential; reading ``a`` as carried too leaves no free group and refuses the split, which is
    exactly the parallel work worth peeling (TSVC ``s222``).
    """
    sdfg, _loop, st = _loop_sdfg('flat_elementwise_rmw', arrays=('a', 'e', 'x', 'y'))
    t_a = st.add_tasklet('_Add_', {'aa', 'xx'}, {'out'}, 'out = aa + xx')
    st.add_edge(st.add_read('a'), None, t_a, 'aa', dace.Memlet('a[i]'))
    st.add_edge(st.add_read('x'), None, t_a, 'xx', dace.Memlet('x[i]'))
    st.add_edge(t_a, 'out', st.add_write('a'), None, dace.Memlet('a[i]'))
    t_e = st.add_tasklet('_Mult_', {'ee', 'yy'}, {'out'}, 'out = ee * yy')
    st.add_edge(st.add_read('e'), None, t_e, 'ee', dace.Memlet('e[i - 1]'))
    st.add_edge(st.add_read('y'), None, t_e, 'yy', dace.Memlet('y[i]'))
    st.add_edge(t_e, 'out', st.add_write('e'), None, dace.Memlet('e[i]'))

    fused = copy.deepcopy(sdfg)
    assert _split(sdfg) == 1, 'the split must peel the elementwise statement off the recurrence'
    loops = _loops(sdfg)
    assert len(loops) == 2
    assert sorted(w for l in loops for w in _written(sdfg, l)) == ['a', 'e']
    sdfg.validate()
    assert_matches_fused(sdfg, fused)


def test_opposite_pulling_constraints_are_refused():
    """``a[i] = b[i+1] + x[i]`` beside ``b[i] = a[i+1] * 2`` -- each group must run first.

    Both reads are one AHEAD of the other group's store, so each wants the ORIGINAL values and each
    rule names the reader. The two constraints form a cycle, and a cycle is not an order: the loop
    stands as it was. Sorting the constraints must not degenerate into picking one of them.
    """
    sdfg, _loop, st = _loop_sdfg('flat_cycle')
    t_a = st.add_tasklet('_Add_', {'bb', 'xx'}, {'out'}, 'out = bb + xx')
    st.add_edge(st.add_read('b'), None, t_a, 'bb', dace.Memlet('b[i + 1]'))
    st.add_edge(st.add_read('x'), None, t_a, 'xx', dace.Memlet('x[i]'))
    st.add_edge(t_a, 'out', st.add_write('a'), None, dace.Memlet('a[i]'))
    t_b = st.add_tasklet('_Mult_', {'aa'}, {'out'}, 'out = aa * 2.0')
    st.add_edge(st.add_read('a'), None, t_b, 'aa', dace.Memlet('a[i + 1]'))
    st.add_edge(t_b, 'out', st.add_write('b'), None, dace.Memlet('b[i]'))

    assert _refuses(sdfg)


def test_scalar_rotation_is_refused():
    """``b[i] = s; s = x[i]`` -- ``s`` holds the PREVIOUS iteration's value.

    Each clone gets a private ``s`` and its own dead-code pruning, so the clone that consumes the
    rotation would lose the statement that produces it.
    """
    sdfg, _loop, st = _loop_sdfg('flat_rotation', arrays=('a', 'b', 'x'))
    sdfg.add_scalar('s', dace.float64, transient=True)
    tb = st.add_tasklet('tb', {'sv'}, {'out'}, 'out = sv')
    st.add_edge(st.add_read('s'), None, tb, 'sv', dace.Memlet('s'))
    st.add_edge(tb, 'out', st.add_write('b'), None, dace.Memlet('b[i]'))
    ts = st.add_tasklet('ts', {'xx'}, {'out'}, 'out = xx')
    st.add_edge(st.add_read('x'), None, ts, 'xx', dace.Memlet('x[i]'))
    st.add_edge(ts, 'out', st.add_write('s'), None, dace.Memlet('s'))
    ta = st.add_tasklet('ta', {'xx'}, {'out'}, 'out = xx * 2.0')
    st.add_edge(st.add_read('x'), None, ta, 'xx', dace.Memlet('x[i]'))
    st.add_edge(ta, 'out', st.add_write('a'), None, dace.Memlet('a[i]'))
    assert _refuses(sdfg)


def test_wcr_output_is_refused():
    """A reduction store is not replicable per group."""
    sdfg, _loop, st = _loop_sdfg('flat_wcr')
    t0 = st.add_tasklet('t0', {'xx'}, {'out'}, 'out = xx')
    st.add_edge(st.add_read('x'), None, t0, 'xx', dace.Memlet('x[i]'))
    st.add_edge(t0, 'out', st.add_write('a'), None, dace.Memlet('a[0]', wcr='lambda p, q: p + q'))
    t1 = st.add_tasklet('t1', {'yy'}, {'out'}, 'out = yy * 2.0')
    st.add_edge(st.add_read('y'), None, t1, 'yy', dace.Memlet('y[i]'))
    st.add_edge(t1, 'out', st.add_write('b'), None, dace.Memlet('b[i]'))
    assert _refuses(sdfg)


def test_data_dependent_trip_count_is_refused():
    """Both loops must sweep the SAME iteration space, so the condition may not read an array."""
    sdfg = dace.SDFG('flat_while')
    for nm in ('a', 'b', 'x', 'y'):
        sdfg.add_array(nm, [N], dace.float64)
    sdfg.add_scalar('g', dace.float64)
    loop = LoopRegion('loop', 'g > 0.0', 'i', 'i = 1', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    st = loop.add_state('body', is_start_block=True)
    t0 = st.add_tasklet('t0', {'xx'}, {'out'}, 'out = xx')
    st.add_edge(st.add_read('x'), None, t0, 'xx', dace.Memlet('x[i]'))
    st.add_edge(t0, 'out', st.add_write('a'), None, dace.Memlet('a[i]'))
    t1 = st.add_tasklet('t1', {'yy'}, {'out'}, 'out = yy * 2.0')
    st.add_edge(st.add_read('y'), None, t1, 'yy', dace.Memlet('y[i]'))
    st.add_edge(t1, 'out', st.add_write('b'), None, dace.Memlet('b[i]'))
    assert _refuses(sdfg)


def test_counter_read_after_the_loop_is_refused():
    """The outlining exports such a counter as an extra scalar output the clones cannot share."""
    sdfg = carry_loop('flat_exported_counter')
    after = sdfg.add_state('after')
    sdfg.add_edge(_loops(sdfg)[0], after, dace.InterstateEdge())
    t = after.add_tasklet('t', {}, {'out'}, 'out = i')
    after.add_edge(t, 'out', after.add_write('b'), None, dace.Memlet('b[0]'))
    assert _refuses(sdfg)


def test_s2710_shaped_guarded_rmw_stays_refused():
    """A guard reading BOTH in-place arrays selects which one an arm updates.

    Duplicated into every clone, the guard is re-evaluated against already-updated data: the
    if-arm's update to ``a`` flips ``a[i] > b[i]`` and the else-arm's store to ``b`` then fires on
    if-arm lanes. The body is not a single state, so the loop path never even starts.
    """
    sdfg = dace.SDFG('flat_guarded_rmw')
    for nm in ('a', 'b', 'x'):
        sdfg.add_array(nm, [N], dace.float64)
    sdfg.add_scalar('g', dace.float64, transient=True)
    loop = LoopRegion('loop', 'i < N', 'i', 'i = 1', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)

    pre = loop.add_state('guard', is_start_block=True)
    tg = pre.add_tasklet('g', {'av', 'bv'}, {'out'}, 'out = 1.0 if av > bv else 0.0')
    pre.add_edge(pre.add_read('a'), None, tg, 'av', dace.Memlet('a[i]'))
    pre.add_edge(pre.add_read('b'), None, tg, 'bv', dace.Memlet('b[i]'))
    pre.add_edge(tg, 'out', pre.add_write('g'), None, dace.Memlet('g'))

    cond = ConditionalBlock('arms')
    loop.add_node(cond)
    loop.add_edge(pre, cond, dace.InterstateEdge())
    then_body = ControlFlowRegion('then_body', sdfg=sdfg)
    st_then = then_body.add_state('t', is_start_block=True)
    ta = st_then.add_tasklet('ta', {'av', 'xv'}, {'out'}, 'out = av + xv')
    st_then.add_edge(st_then.add_read('a'), None, ta, 'av', dace.Memlet('a[i]'))
    st_then.add_edge(st_then.add_read('x'), None, ta, 'xv', dace.Memlet('x[i]'))
    st_then.add_edge(ta, 'out', st_then.add_write('a'), None, dace.Memlet('a[i]'))
    cond.add_branch(dace.properties.CodeBlock('g > 0.0'), then_body)
    else_body = ControlFlowRegion('else_body', sdfg=sdfg)
    st_else = else_body.add_state('e', is_start_block=True)
    tb = st_else.add_tasklet('tb', {'bv'}, {'out'}, 'out = bv * 2.0')
    st_else.add_edge(st_else.add_read('b'), None, tb, 'bv', dace.Memlet('b[i]'))
    st_else.add_edge(tb, 'out', st_else.add_write('b'), None, dace.Memlet('b[i]'))
    cond.add_branch(None, else_body)

    assert _refuses(sdfg)


def test_single_output_loop_is_left_alone():
    sdfg, _loop, st = _loop_sdfg('flat_one_output', arrays=('a', 'x'))
    t0 = st.add_tasklet('t0', {'xx'}, {'out'}, 'out = xx * 2.0')
    st.add_edge(st.add_read('x'), None, t0, 'xx', dace.Memlet('x[i]'))
    st.add_edge(t0, 'out', st.add_write('a'), None, dace.Memlet('a[i]'))
    assert _refuses(sdfg)


# ===========================================================================
# End to end: the pipeline outcome does not regress now that the split, and no
# longer LoopFission, is what distributes this loop.
# ===========================================================================
def test_canonicalize_still_parallelizes_the_motivating_kernel():
    """Canonicalization ends with no loops left and the parallel statement in a map."""
    from dace.transformation.passes.canonicalize.pipeline import canonicalize

    @dace.program
    def kern(a: dace.float64[N], b: dace.float64[N], x: dace.float64[N], y: dace.float64[N]):
        for i in range(1, N):
            s1 = x[i]
            a[i] = a[i - 1] + s1
            b[i] = y[i] * 2.0

    n = 32
    rng = np.random.default_rng(7)
    base = {'a': rng.random(n), 'b': np.zeros(n), 'x': rng.random(n), 'y': rng.random(n)}
    ref = {k: v.copy() for k, v in base.items()}
    kern.to_sdfg(simplify=True).compile()(**ref, N=n)

    sdfg = kern.to_sdfg(simplify=True)
    canonicalize(sdfg)
    assert not _loops(sdfg)
    assert [n for s in sdfg.all_states() for n in s.nodes() if isinstance(n, nodes.MapEntry)]
    got = {k: v.copy() for k, v in base.items()}
    sdfg.compile()(**got, N=n)
    assert np.allclose(got['a'], ref['a'])
    assert np.allclose(got['b'], ref['b'])


def test_split_loops_knob_off_leaves_the_loop_alone():
    sdfg = carry_loop('flat_knob_off')
    before = sdfg.hash_sdfg()
    assert SplitStatements(split_loops=False).apply_pass(sdfg, {}) is None
    assert sdfg.hash_sdfg() == before


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
