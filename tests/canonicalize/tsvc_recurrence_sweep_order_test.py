# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The recurrence-sweep loop ORDER, asserted on what ``canonicalize`` emits.

``aa[j, i] = aa[j - 1, i] + ...`` has one carried axis (``j``) and one independent axis (``i``),
and ``i`` is the contiguous one. Only one of the two orders can be innermost, and the choice is
made by :mod:`~dace.transformation.passes.canonicalize.move_loop_into_map_gated`. Putting the
independent axis outermost buys outer parallelism and pays for it with a ``stride = LEN_2D``
innermost access -- one cache line and one TLB entry per element, and nothing for the vectorizer
to strip-mine. The pass used to take that trade unconditionally whenever the map's axis was
unit-stride; it no longer does.

What is asserted here is the ORDER and the innermost STRIDE, never a wall clock: the order is the
property the cost model decides, and it is readable off the graph. The numeric check rides along
because an order assertion alone passes just as happily on a miscompiled nest.
"""
import os

os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.passes.canonicalize.move_loop_into_map_gated import (MoveLoopIntoMapGated,
                                                                              interchange_lowers_stride)
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.minimize_stride_permutation import _to_float, score_indexed_strides

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES


def canonicalized(name, tag):
    """``(kernel, sdfg)`` for one TSVC kernel put through the production canonicalize recipe."""
    kernel = tsvc.collect(name=name)[0]
    sdfg = tsvc.to_sdfg(kernel, tag, simplify=True)
    canonicalize(sdfg, validate=True)
    return kernel, sdfg


def iteration_order(sdfg):
    """The iterated axes of ``sdfg``, outermost first, crossing loops, maps and nested SDFGs.

    A ``LoopRegion`` contributes its loop variable and a map scope contributes its parameters, so
    the list reads as the loop nest the emitter will write. Only nests with a single spine are
    described; a graph that forks into two independent nests yields both spines concatenated,
    which is why every consumer here asserts on a kernel known to have one.
    """
    order = []

    def walk_state(state, entry):
        for node in state.nodes():
            if isinstance(node, nd.MapEntry) and state.entry_node(node) is entry:
                order.extend(node.map.params)
                walk_state(state, node)
            elif isinstance(node, nd.NestedSDFG) and state.entry_node(node) is entry:
                walk_region(node.sdfg)

    def walk_region(region):
        for block in region.nodes():
            if isinstance(block, LoopRegion):
                if block.loop_variable:
                    order.append(str(block.loop_variable))
                walk_region(block)
            elif isinstance(block, ConditionalBlock):
                # A guarded loop is still an iterated axis; its branches hold the blocks, so a
                # plain ControlFlowRegion walk would step straight past the nest inside a guard.
                for _cond, branch in block.branches:
                    walk_region(branch)
            elif isinstance(block, ControlFlowRegion):
                walk_region(block)
            elif isinstance(block, SDFGState):
                walk_state(block, None)

    walk_region(sdfg)
    return order


def innermost_stride(sdfg, array):
    """The array stride the INNERMOST iterated axis walks in ``array``.

    ``1`` means the emitted inner loop is contiguous. Scored with the same ranker the cost model
    uses, so the test measures the property the pass claims to optimize rather than a proxy.
    """
    axis = iteration_order(sdfg)[-1]
    best = float('inf')
    for sub in sdfg.all_sdfgs_recursive():
        if array not in sub.arrays:
            continue
        for state in sub.states():
            scores = score_indexed_strides(state.edges(), sub, [axis])
            if axis in scores:
                best = min(best, _to_float(scores[axis][0]))
    return best


def carried_axis(sdfg):
    """The axis of the one sequential ``LoopRegion`` canonicalize left behind."""
    loops = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion) and r.loop_variable]
    assert len(loops) == 1, f'expected exactly one residual sequential loop, got {[r.label for r in loops]}'
    return str(loops[0].loop_variable)


def assert_matches_reference(kernel, sdfg):
    """The canonicalized kernel must reproduce the numpy reference element for element.

    Structure is read off ``sdfg`` BEFORE this runs: compiling is one-way, and the SDFG a
    ``CompiledSDFG`` still points at is not the object to inspect afterwards.
    """
    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[kernel.name](**ref, **call_kwargs)
    got = {n: a.copy() for n, a in arrays.items()}
    sdfg.compile()(**got, **call_kwargs)
    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue
        assert np.allclose(ref[name], got[name], equal_nan=True), f'{kernel.name}: value mismatch on {name}'


@pytest.mark.parametrize('name', ['s231_d_single', 's235_d_single'])
def test_carried_axis_is_outermost_and_the_contiguous_axis_is_innermost(name):
    """``aa[j, i] = aa[j-1, i] + ...``: ``j`` carries, ``i`` is contiguous, so ``i`` goes inside.

    The defect this pins is the opposite order -- ``i`` outermost and parallel, ``j`` inside at
    ``stride = LEN_2D``. Both orders run the same iterations; only one of them has a contiguous
    innermost access, and that is the one the cost model must pick.
    """
    kernel, sdfg = canonicalized(name, 'sweep_order_' + name)
    order = iteration_order(sdfg)
    carried = carried_axis(sdfg)
    assert order[-1] != carried, f'{name}: the carried axis {carried} must not be innermost, got {order}'
    assert order.index(carried) < len(order) - 1, f'{name}: {carried} must enclose the independent axis, got {order}'
    assert_matches_reference(kernel, sdfg)


@pytest.mark.parametrize('name', ['s231_d_single', 's235_d_single'])
def test_innermost_access_is_unit_stride(name):
    """The emitted inner loop must walk ``aa`` contiguously -- that is the whole point of the order."""
    _kernel, sdfg = canonicalized(name, 'sweep_stride_' + name)
    assert innermost_stride(sdfg, 'aa') == 1.0, \
        f'{name}: innermost axis {iteration_order(sdfg)[-1]} does not walk aa with stride 1'


@pytest.mark.parametrize('name', ['s1232_d_single', 's2275_d_single'])
def test_fully_parallel_nests_keep_their_contiguous_inner_axis(name):
    """The nests where the interchange already fired must not regress.

    Neither kernel carries a dependence, so both axes become maps and the loop<->map gate is never
    consulted; the ordering comes from ``MinimizeStridePermutation``. Asserted here so a change to
    the gate that accidentally reaches these shows up as a failure rather than as a slowdown.
    """
    kernel, sdfg = canonicalized(name, 'sweep_par_' + name)
    assert innermost_stride(sdfg, 'aa') == 1.0, f'{name}: innermost axis must walk aa with stride 1'
    assert_matches_reference(kernel, sdfg)


def test_gate_declining_leaves_the_sdfg_untouched():
    """A graph the gate declines must come out bit-identical -- no partial rewrite, no renaming."""
    kernel = tsvc.collect(name='s231_d_single')[0]
    sdfg = tsvc.to_sdfg(kernel, 'sweep_noop', simplify=True)
    canonicalize(sdfg, validate=True)
    before = sdfg.to_json()
    assert MoveLoopIntoMapGated(target='cpu').apply_pass(sdfg, {}) is None, \
        'the gate must decline the settled recurrence sweep'
    assert sdfg.to_json() == before, 'a declined gate must not mutate the SDFG'


def test_gate_declines_when_the_map_axis_is_the_contiguous_one():
    """The cost model's own predicate, read directly rather than inferred from the emitted code.

    ``interchange_lowers_stride`` is the whole CPU rule: hoist only when the LOOP variable is the
    more contiguous of the two. On the recurrence sweep the map's ``i`` is contiguous and the
    loop's ``j`` is not, so it must answer False -- and, because it is now the only rule, the
    interchange is declined.
    """
    from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap

    @dace.program
    def sweep(aa: dace.float64[64, 64], bb: dace.float64[64, 64]):
        for i in dace.map[0:64]:
            aa[0, i] = bb[0, i]
        for j in range(1, 64):
            for i in dace.map[0:64]:
                aa[j, i] = aa[j - 1, i] + bb[j, i]

    sdfg = sweep.to_sdfg(simplify=True)
    loops = [
        r for r in sdfg.all_control_flow_regions(recursive=True)
        if isinstance(r, LoopRegion) and r.loop_variable and MoveLoopIntoMap.can_be_applied_to(r.sdfg, loop=r)
    ]
    assert loops, 'the fixture must present an interchangeable loop<->map pair'
    for loop in loops:
        assert not interchange_lowers_stride(loop, loop.sdfg), \
            f'{loop.loop_variable} is the strided axis; hoisting its map must not be judged a win'


@pytest.mark.xfail(strict=True,
                   reason='LoopStridePermutation._perfect_nests requires each level to hold exactly ONE block, and '
                   "s275's i-loop holds two: the guard prep aa_index = aa[0, i] and the j-loop. MoveIfIntoLoop "
                   'sinks the guard itself into the j-loop but leaves the i-dependent prep behind, so the nest '
                   'never becomes perfect and the interchange is never offered. Needs the prep hoisted to a mask '
                   'materialized once (legal here because aa[0, i] is read before any write -- j starts at 1), '
                   'which is a transformation that does not exist yet.')
def test_s275_guarded_sweep_gets_the_same_order():
    """``if aa[0, i] > 0: for j: aa[j, i] = aa[j-1, i] + ...`` -- same sweep, behind a guard.

    The guard is invariant in ``j`` and its operand is never written by the nest, so the same
    order is legal here. Diagnosed and not yet reachable; see the xfail reason for the predicate
    that refuses it.
    """
    _kernel, sdfg = canonicalized('s275_d_single', 'sweep_guard')
    order = iteration_order(sdfg)
    carried = carried_axis(sdfg)
    assert order[-1] != carried, f's275: the carried axis {carried} must not be innermost, got {order}'
