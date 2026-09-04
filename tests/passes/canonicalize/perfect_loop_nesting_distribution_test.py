# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PerfectLoopNesting`` at NEST granularity.

The corpus half covers the seven TSVC-2 kernels whose outer loop body holds more than one
top-level child, found by an AST sweep:

* SIBLING inner loops -- ``s2233``, ``s233``. The two children carry in OPPOSITE directions, so
  under one parent neither level is parallel for the body as a whole: ``i`` is blocked by the
  second child and ``j`` by the first. Distributed, each child gets its own level back.
* IMPERFECT nest, a loop beside a bare statement -- ``s2275``, ``s235``, ``s2102``, ``s141``,
  ``s126``. Four of these distribute into ``for i { s1 }`` beside a PERFECT ``for i { for j { s2 } }``
  (two directly, two once the pipeline's IV passes close their counter); only ``s2102`` is a
  genuine legality refusal.

The synthetic half covers what TSVC-2 does not contain: an adversarial cross-child dependence, a
footprint the analysis cannot pin, more than two children, and a component that straddles an
independent sibling.

The pass takes the finest LEGAL partition unconditionally -- there is no profitability gate, and
the downstream fusion stages recombine what the split did not pay for. What is asserted here is
STRUCTURE -- how many nests, which of their levels are parallel -- plus the values. Never a
speedup.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes, utils as sdutil
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.canonicalize.distribute_producer_consumer import _forward_flow_groups
from dace.transformation.passes.canonicalize.hoist_iv_updates import HoistInductionVariableUpdates
from dace.transformation.passes.canonicalize.induction_variable_substitution import InductionVariableSubstitution
from dace.transformation.passes.canonicalize.perfect_loop_nesting import (PerfectLoopNesting, distribute_loops,
                                                                          level_parallel, parallel_level_diagnostic)
from dace.transformation.passes.loop_fission import LoopFission, _linear_blocks
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES

#: The two kernels whose outer loop body holds sibling inner loops carrying in opposite directions.
SIBLING_NESTS = ['s2233', 's233']

#: Imperfect nests whose two children ARE separable, though the parent level was already parallel
#: for both -- legal, so they distribute; the fusion stages are what may put them back.
SEPARABLE_IMPERFECT = ['s2275', 's235']

#: Imperfect nests whose body edge carries a cross-loop induction variable. Whether they
#: distribute is decided by whether the pipeline's IV passes have closed that counter yet, which is
#: read off the SDFG rather than assumed -- see :func:`test_induction_variable_decides_the_verdict`.
INDUCTION_VARIABLE_NESTS = ['s126', 's141']

#: The one genuine legality refusal in the corpus.
REFUSED_NESTS = {
    's2102': 'both children WRITE aa -- one dependence component',
}

#: ``kernel -> (fused, distributed)`` parallel levels AT THE PARENT LEVEL, reported by
#: :func:`parallel_level_diagnostic` for the split the pass performs. A diagnostic, not a gate:
#: ``s2233`` / ``s233`` gain a level, ``s2275`` / ``s235`` do not, and the pass splits all four.
PARENT_LEVEL_DIAGNOSTIC = {
    's2233': (0, 1),
    's233': (0, 1),
    's2275': (1, 2),
    's235': (1, 2),
}

#: ``kernel -> ((parallel, total), (parallel, total))`` loop-nest levels of the WHOLE kernel,
#: fused against distributed, once every liftable loop has become a Map. The two induction-variable
#: kernels are absent because their answer depends on the IV passes, not on this one.
WHOLE_KERNEL_LEVELS = {
    's2233': ((1, 3), (2, 4)),
    's233': ((1, 3), (2, 4)),
    's2275': ((2, 2), (3, 3)),
    's235': ((1, 2), (2, 3)),
    's2102': ((2, 2), (2, 2)),
}

N = dace.symbol('N')


def build(name, tag):
    """The corpus kernel ``name`` as a fresh, uniquely-named SDFG."""
    kernel = [k for k in tsvc.collect() if k.name == name + '_d_single'][0]
    return kernel, tsvc.to_sdfg(kernel, tag, simplify=True)


def nest_levels(sdfg):
    """``(parallel, total)`` loop-nest levels: a Map parameter is a parallel level, a
    ``LoopRegion`` counter a sequential one."""
    parallel = sum(len(n.map.params) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    sequential = sum(1 for r in sdfg.all_control_flow_regions(recursive=True)
                     if isinstance(r, LoopRegion) and r.loop_variable)
    return parallel, parallel + sequential


def outer_kinds(sdfg):
    """``'map'`` / ``'loop'`` for the OUTERMOST level of every top-level nest, in block order."""
    kinds = []
    for block in sdfg.nodes():
        if isinstance(block, LoopRegion):
            kinds.append('loop')
        elif isinstance(block, SDFGState) and any(
                isinstance(n, nodes.MapEntry) and block.entry_node(n) is None for n in block.nodes()):
            kinds.append('map')
    return kinds


def top_level_nests(sdfg):
    """How many separate nests the SDFG's top level holds."""
    return len(outer_kinds(sdfg))


def parallelize(sdfg):
    """Lift every loop the dependence analysis clears into a Map, and return ``sdfg``."""
    PatternMatchAndApplyRepeated([LoopToMap()]).apply_pass(sdfg, {})
    return sdfg


def only_loop(sdfg):
    """The SDFG's single top-level ``LoopRegion``."""
    return next(r for r in sdfg.nodes() if isinstance(r, LoopRegion))


def body_blocks(loop):
    """``loop``'s body blocks as a plain chain -- the granularity the distribution works at."""
    blocks = _linear_blocks(loop)
    assert blocks is not None, 'the body is not a plain chain, so nothing here applies'
    return blocks


def is_perfect_nest(loop):
    """Whether ``loop``'s body is exactly one block -- the perfect nest the later phases need, and
    the only shape whose two levels collapse into one 2-D map."""
    return len(loop.nodes()) == 1


def has_open_induction_variable(sdfg):
    """Whether any loop body edge still ASSIGNS -- an induction variable the IV passes have not
    closed. Observable, so a test never has to assume which state the tree is in."""
    return any(edge.data.assignments for region in sdfg.all_control_flow_regions(recursive=True)
               if isinstance(region, LoopRegion) for edge in region.edges())


def close_induction_variables(sdfg):
    """Run the pipeline's IV-closing passes ahead of the distribution, exactly as the 'reduce' stage
    does before 'fission'."""
    SimplifyPass().apply_pass(sdfg, {})
    HoistInductionVariableUpdates().apply_pass(sdfg, {})
    InductionVariableSubstitution().apply_pass(sdfg, {})
    return sdfg


def written_arrays(block):
    """The non-transient containers ``block`` writes, in first-seen order."""
    states = list(block.all_states()) if isinstance(block, LoopRegion) else [block]
    names = []
    for state in states:
        for node in state.data_nodes():
            if state.in_degree(node) > 0 and not state.sdfg.arrays[node.data].transient and node.data not in names:
                names.append(node.data)
    return names


def emitted_write_order(sdfg):
    """What each distributed nest writes, in EXECUTION order -- the emitted component order."""
    return [
        written_arrays(block) for block in sdutil.dfs_topological_sort(sdfg, [sdfg.start_block])
        if isinstance(block, LoopRegion)
    ]


def run_matches_reference(kernel, sdfg):
    """Whether ``sdfg`` reproduces the kernel's NumPy reference on one seeded input set."""
    arrays, call_kwargs = tsvc.make_inputs(kernel)
    expected = {name: arr.copy() for name, arr in arrays.items()}
    REFERENCES[kernel.name](**expected, **call_kwargs)
    got = {name: arr.copy() for name, arr in arrays.items()}
    sdfg.compile()(**got, **call_kwargs)
    return all(np.allclose(expected[name], got[name], equal_nan=True) for name in arrays)


def run_matches_untransformed(program, transformed, size=6, seed=11):
    """Whether ``transformed`` still computes what the untransformed program computes."""
    rng = np.random.default_rng(seed)
    names = [
        name for name, desc in program.to_sdfg(simplify=True).arglist().items() if isinstance(desc, dace.data.Array)
    ]
    inputs = {name: rng.random((size, size)) for name in names}

    reference = program.to_sdfg(simplify=True)
    expected = {name: arr.copy() for name, arr in inputs.items()}
    reference(**expected, N=size)
    got = {name: arr.copy() for name, arr in inputs.items()}
    transformed(**got, N=size)
    return all(np.allclose(expected[name], got[name]) for name in names)


# --------------------------------------------------------------------------------------------
# The sibling-nest shape: distribution is the only way to a second parallel level.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize('name', SIBLING_NESTS)
def test_fused_sibling_nests_have_one_parallel_level_of_three(name):
    """Undistributed, the shared parent is blocked by the second child and the first child's own
    inner level by its own recurrence: of the three levels only one parallelizes."""
    _kernel, sdfg = build(name, f'fused_{name}')
    assert nest_levels(parallelize(sdfg)) == (1, 3)
    assert top_level_nests(sdfg) == 1, 'the fused form is one nest'


@pytest.mark.parametrize('name', SIBLING_NESTS)
def test_sibling_nests_distribute_into_two_nests(name):
    """``PerfectLoopNesting`` splits the parent into one loop per dependence component, in program
    order: the two inner loops become two separate nests.

    This kernel says little about LEGALITY -- the children share only ``cc``, which both READ, so
    there is no cross-child dependence for any rule to get wrong. The adversarial nests further
    down are what decide the legality question."""
    _kernel, sdfg = build(name, f'split_{name}')
    assert PerfectLoopNesting().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    assert top_level_nests(sdfg) == 2
    assert nest_levels(sdfg) == (0, 4), 'four sequential levels before any lift'


@pytest.mark.parametrize('name', SIBLING_NESTS)
def test_distribution_frees_a_second_parallel_level(name):
    """After the split each child parallelizes on the level the other one blocked: one nest becomes
    ``map i { loop j }``, the other ``loop i { map j }`` -- two parallel levels of four, against
    the one of three the fused nest reaches."""
    _kernel, sdfg = build(name, f'levels_{name}')
    PerfectLoopNesting().apply_pass(sdfg, {})
    parallelize(sdfg)
    sdfg.validate()
    assert nest_levels(sdfg) == (2, 4)
    assert sorted(outer_kinds(sdfg)) == ['loop', 'map'], \
        'one nest must be outer-parallel and the other outer-sequential'


@pytest.mark.parametrize('name', SIBLING_NESTS + SEPARABLE_IMPERFECT)
def test_distributed_kernels_preserve_values(name):
    """The split is only worth having if it is also correct."""
    kernel, sdfg = build(name, f'values_{name}')
    assert PerfectLoopNesting().apply_pass(sdfg, {}) is not None
    parallelize(sdfg)
    sdfg.validate()
    assert run_matches_reference(kernel, sdfg)


# --------------------------------------------------------------------------------------------
# Legality alone decides: a separable pair splits even where it frees no parent level.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize('name', SEPARABLE_IMPERFECT)
def test_separable_imperfect_nests_distribute_though_they_free_no_level(name):
    """``s2275`` / ``s235`` separate legally, and the parent level was already parallel for BOTH
    children, so the split frees nothing at that level. It happens anyway: canonicalization takes
    the finest legal partition and leaves the recombining to the fusion stages, because a
    "distribute only when it pays" rule is a cost decision and there is no cost model here."""
    _kernel, sdfg = build(name, f'separable_{name}')
    loop = only_loop(sdfg)
    blocks = body_blocks(loop)
    assert level_parallel(blocks, loop.loop_variable, loop.sdfg.arrays), \
        'the fused parent level is already parallel for the whole body'
    assert PerfectLoopNesting().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    assert top_level_nests(sdfg) == 2
    assert all(is_perfect_nest(r) for r in sdfg.nodes() if isinstance(r, LoopRegion)), \
        'each product must be a PERFECT nest -- that is the shape the later phases collapse'


@pytest.mark.parametrize('name', sorted(PARENT_LEVEL_DIAGNOSTIC))
def test_parallel_level_diagnostic_is_reported_per_split(name):
    """The parallel-level comparison survives as a REPORTED number, not a decision. Recorded per
    split so a sweep can see which kernels a distribution actually freed a level for."""
    _kernel, sdfg = build(name, f'diag_{name}')
    loop = only_loop(sdfg)
    groups = _forward_flow_groups(loop)
    assert groups is not None
    assert parallel_level_diagnostic(loop, groups) == PARENT_LEVEL_DIAGNOSTIC[name]

    records = []
    assert distribute_loops(sdfg, records) == 1
    assert [(fused, distributed) for _label, fused, distributed in records] == [PARENT_LEVEL_DIAGNOSTIC[name]]


@pytest.mark.parametrize('name', sorted(WHOLE_KERNEL_LEVELS))
def test_whole_kernel_levels_fused_against_distributed(name):
    """The corpus scorecard: parallel and total loop-nest levels before and after the pass, for the
    five kernels whose answer this pass alone decides. Structure only -- no timing claim is made or
    implied."""
    fused_expected, split_expected = WHOLE_KERNEL_LEVELS[name]
    _kernel, fused = build(name, f'wkfused_{name}')
    assert nest_levels(parallelize(fused)) == fused_expected

    _kernel, split = build(name, f'wksplit_{name}')
    PerfectLoopNesting().apply_pass(split, {})
    parallelize(split)
    split.validate()
    assert nest_levels(split) == split_expected


@pytest.mark.parametrize('name', sorted(WHOLE_KERNEL_LEVELS))
def test_pass_is_idempotent(name):
    """The finest legal partition is a fixpoint: a second application changes nothing."""
    _kernel, sdfg = build(name, f'idem_{name}')
    PerfectLoopNesting().apply_pass(sdfg, {})
    once = sdfg.to_json()
    PerfectLoopNesting().apply_pass(sdfg, {})
    assert sdfg.to_json() == once


# --------------------------------------------------------------------------------------------
# The refused nests: a pass that refuses changes nothing.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize('name', sorted(REFUSED_NESTS))
def test_refused_nests_are_byte_identical(name):
    """A refusal must leave the SDFG exactly as it was."""
    _kernel, sdfg = build(name, f'refuse_{name}')
    before = sdfg.to_json()
    assert PerfectLoopNesting().apply_pass(sdfg, {}) is None, REFUSED_NESTS[name]
    assert sdfg.to_json() == before, 'a refusal mutated the SDFG'


def test_shared_written_array_is_one_dependence_component():
    """``s2102`` zeroes a column and then writes the diagonal element of that same column. Both
    children write ``aa``, so they form ONE component and the loop stands -- the subset-disjointness
    reasoning that would separate them is out of this pass's scope."""
    _kernel, sdfg = build('s2102', 'component_s2102')
    loop = only_loop(sdfg)
    assert _linear_blocks(loop) is not None, 'the body IS a plain two-block chain'
    assert _forward_flow_groups(loop) is None, 'but the two blocks are one dependence component'


@pytest.mark.parametrize('name', INDUCTION_VARIABLE_NESTS)
def test_induction_variable_decides_the_verdict(name):
    """``s126``'s ``k`` is incremented once per inner iteration AND once per outer iteration;
    ``s141`` reseeds ``k`` from ``i`` at the top of the body. While that counter sits on an
    interstate edge inside the body, ``flat_2d_array[k - 1]`` is not an affine function of the
    iteration and cloning the loop would run the increment once per clone -- so the body is refused
    for carrying an assignment at all, before any dependence question is asked. Once
    ``InductionVariableSubstitution`` closes ``k`` the blocker is gone and the nest distributes like
    any other.

    Which of the two states the tree is in is READ OFF THE SDFG rather than assumed: the IV passes
    are another module's, and this expectation must not depend on the order two people finished in.
    Values are checked in either state."""
    kernel, raw = build(name, f'ivraw_{name}')
    assert has_open_induction_variable(raw), 'the counter starts on a body edge'
    before = raw.to_json()
    assert PerfectLoopNesting().apply_pass(raw, {}) is None, 'an open counter blocks the split'
    assert raw.to_json() == before

    kernel, sdfg = build(name, f'ivclosed_{name}')
    close_induction_variables(sdfg)
    still_open = has_open_induction_variable(sdfg)
    applied = PerfectLoopNesting().apply_pass(sdfg, {})
    sdfg.validate()
    if still_open:
        assert applied is None, 'the counter survived, so the body still carries an assignment'
    else:
        assert applied is not None, 'the counter is closed, so the nest must distribute'
        assert all(is_perfect_nest(r) for r in sdfg.nodes() if isinstance(r, LoopRegion))
    assert run_matches_reference(kernel, sdfg)


# --------------------------------------------------------------------------------------------
# Synthetic nests: a child is a whole footprint per parent iteration, not a point.
# --------------------------------------------------------------------------------------------


@dace.program
def backward_carried(x: dace.float64[N, N], t: dace.float64[N, N], b: dace.float64[N, N]):
    """Child A READS ``x`` one PARENT iteration behind what child B WRITES.

    Fused, ``A(i)`` sees the column ``B(i - 1)`` just updated. Distributed, A runs to completion
    first and sees the ORIGINAL ``x`` throughout -- a dependence that runs BACKWARDS in the emitted
    order, so the two children may not be separated."""
    for i in range(1, N):
        for j in range(N):
            t[j, i] = x[j, i - 1] * 2.0
        for j in range(N):
            x[j, i] = b[j, i] + 1.0


@dace.program
def range_footprint(x: dace.float64[N, N], t: dace.float64[N], b: dace.float64[N, N]):
    """Child A writes a whole column SLICE per parent iteration. The footprint is a range, not an
    element pinned to ``i``, so nothing separates one parent iteration's writes from another's and
    the pair must stay together."""
    for i in range(N):
        x[:, i] = b[:, i] * 2.0
        for j in range(N):
            t[i] = t[i] + x[j, i]


@dace.program
def per_iteration_producer(x: dace.float64[N, N], t: dace.float64[N, N], b: dace.float64[N, N]):
    """Child A writes ``x[j, i]`` and child B reads exactly ``x[j, i]``: both pinned to the parent
    index, so each parent iteration owns its own column and A(i) is still the only writer B(i) can
    see after the split."""
    for i in range(N):
        for j in range(N):
            x[j, i] = b[j, i] + 1.0
        for j in range(N):
            t[j, i] = x[j, i] * 2.0


@dace.program
def chain_plus_independent(p: dace.float64[N, N], q: dace.float64[N, N], r: dace.float64[N, N], s: dace.float64[N, N],
                           a: dace.float64[N, N]):
    """FOUR children: an independent one FIRST, then a producer chain ``p -> q -> r``. Every
    dependence points forward and is per-iteration aligned, so all four separate."""
    for i in range(N):
        for j in range(N):
            s[j, i] = a[j, i] * 4.0
        for j in range(N):
            p[j, i] = a[j, i] + 1.0
        for j in range(N):
            q[j, i] = p[j, i] * 2.0
        for j in range(N):
            r[j, i] = q[j, i] + 3.0


@dace.program
def straddled_component(x: dace.float64[N, N], t: dace.float64[N, N], s: dace.float64[N, N], a: dace.float64[N, N]):
    """A component that STRADDLES an independent sibling: the first child reads ``x`` one parent
    iteration behind, the THIRD writes it, and the second is independent of both."""
    for i in range(1, N):
        for j in range(N):
            t[j, i] = x[j, i - 1] * 2.0
        for j in range(N):
            s[j, i] = a[j, i] * 4.0
        for j in range(N):
            x[j, i] = a[j, i] + 1.0


def test_parent_carried_cross_child_read_is_refused():
    """The adversarial case, and the one TSVC-2 does not contain.

    Child B writes an array child A reads one parent iteration behind, so the dependence runs
    backwards in the emitted order. A rule that merged only on a shared WRITTEN container would
    separate these two and be wrong -- a shared READ of what a sibling WRITES is a real
    dependence."""
    sdfg = backward_carried.to_sdfg(simplify=True)
    loop = only_loop(sdfg)
    blocks = body_blocks(loop)
    assert len(blocks) == 2
    assert not level_parallel(blocks, loop.loop_variable, loop.sdfg.arrays), 'the fused level is carried'
    assert all(level_parallel([block], loop.loop_variable, loop.sdfg.arrays) for block in blocks), \
        'each child alone is parent-parallel -- the diagnostic would call this split a gain'
    assert _forward_flow_groups(loop) is None, 'but the cross-child dependence runs backwards'

    before = sdfg.to_json()
    assert PerfectLoopNesting().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before


def test_refused_split_would_have_changed_the_values():
    """The refusal is load-bearing, not incidental: performing it anyway changes ``t``.

    Distribution is applied here by hand, past the legality check, purely to show what the check is
    buying. ``x`` still comes out right -- only the reader diverges, which is what a backward
    cross-child dependence looks like when it is ignored."""
    size = 6
    rng = np.random.default_rng(3)
    x0, b0 = rng.random((size, size)), rng.random((size, size))

    reference = backward_carried.to_sdfg(simplify=True)
    ref_x, ref_t, ref_b = x0.copy(), np.zeros((size, size)), b0.copy()
    reference(x=ref_x, t=ref_t, b=ref_b, N=size)

    forced = backward_carried.to_sdfg(simplify=True)
    loop = only_loop(forced)
    LoopFission._fission_blocks(loop, [[block] for block in body_blocks(loop)])
    forced.validate()
    got_x, got_t, got_b = x0.copy(), np.zeros((size, size)), b0.copy()
    forced(x=got_x, t=got_t, b=got_b, N=size)

    assert np.allclose(got_x, ref_x), 'the writer is unaffected either way'
    assert not np.allclose(got_t, ref_t), 'the reader MUST diverge -- otherwise this pins nothing'


def test_over_approximated_footprint_is_refused():
    """A child whose per-parent-iteration footprint is a RANGE cannot be shown disjoint from the
    next iteration's, so the pair stays together. The over-approximation refuses; it never assumes
    disjointness."""
    sdfg = range_footprint.to_sdfg(simplify=True)
    loop = only_loop(sdfg)
    assert _linear_blocks(loop) is not None, 'the body IS a plain two-block chain'
    assert _forward_flow_groups(loop) is None

    before = sdfg.to_json()
    assert PerfectLoopNesting().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before


def test_per_iteration_producer_consumer_distributes_and_preserves_values():
    """A child READS what the other WRITES and the split is still legal, because both accesses are
    pinned to the parent index: each parent iteration owns its own column, so the consumer sees the
    producer's finished values either way."""
    sdfg = per_iteration_producer.to_sdfg(simplify=True)
    loop = only_loop(sdfg)
    groups = _forward_flow_groups(loop)
    assert groups is not None and len(groups) == 2

    assert PerfectLoopNesting().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    assert top_level_nests(sdfg) == 2
    assert run_matches_untransformed(per_iteration_producer, sdfg)


def test_four_children_split_in_program_order():
    """More than two children, which the corpus does not have. All four components separate, and
    the emitted order is the ORIGINAL block order -- pinned, because an unpinned choice among
    mutually incomparable components would only show up later as a seed-dependent difference."""
    sdfg = chain_plus_independent.to_sdfg(simplify=True)
    loop = only_loop(sdfg)
    groups = _forward_flow_groups(loop)
    assert groups is not None and [len(g) for g in groups] == [1, 1, 1, 1]
    assert [written_arrays(block) for group in groups for block in group] == [['s'], ['p'], ['q'], ['r']]

    assert PerfectLoopNesting().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    assert emitted_write_order(sdfg) == [['s'], ['p'], ['q'], ['r']]
    assert run_matches_untransformed(chain_plus_independent, sdfg)


def test_component_straddling_an_independent_sibling():
    """The first and THIRD children are one component -- the third writes what the first reads a
    parent iteration behind -- while the second is independent of both. The partition is over
    components, not over adjacent pairs, so the independent middle child is emitted AFTER the
    component that straddles it. Nothing depended on it being in between; the values prove it."""
    sdfg = straddled_component.to_sdfg(simplify=True)
    loop = only_loop(sdfg)
    groups = _forward_flow_groups(loop)
    assert groups is not None
    assert [[written_arrays(block) for block in group] for group in groups] == [[['t'], ['x']], [['s']]]

    assert PerfectLoopNesting().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    assert emitted_write_order(sdfg) == [['t', 'x'], ['s']]
    assert run_matches_untransformed(straddled_component, sdfg)


def test_distribution_order_is_stable_across_runs():
    """Determinism: the same input gives the same emitted component order every time. Ties among
    incomparable components resolve to program order, never to a set's iteration order."""
    orders = []
    for _ in range(3):
        sdfg = chain_plus_independent.to_sdfg(simplify=True)
        distribute_loops(sdfg)
        UniqueLoopIterators(assign_loop_iterator_post_value=False).apply_pass(sdfg, {})
        orders.append(emitted_write_order(sdfg))
    assert orders[0] == orders[1] == orders[2] == [['s'], ['p'], ['q'], ['r']]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
