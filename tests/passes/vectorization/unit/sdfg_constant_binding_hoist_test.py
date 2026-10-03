# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``BranchNormalization`` hoists a symbol binding that has one value for the whole SDFG out of its arm.

CloudSC's fused riming + melting map binds ``imelt_index = imelt[0]`` in the middle of an arm nested
under a per-column guard, and a second arm binds the same value for its own read. Left inside the
guard the binding cannot be tiled (one symbol, many lanes) and the vectorizer refused all of CloudSC.
"""
import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization


def add_write(state: dace.SDFGState, target: str, code: str, source: str = None) -> None:
    tasklet = state.add_tasklet('t', ['x'] if source else [], ['y'], code)
    if source:
        state.add_edge(state.add_read(source.split('[')[0]), None, tasklet, 'x', dace.Memlet(source))
    state.add_edge(tasklet, 'y', state.add_write(target.split('[')[0]), None, dace.Memlet(target))


def guarded_arm(label: str, condition: str, out: str, binding: str) -> ConditionalBlock:
    """``if condition: out = 0; k = binding; out = z[k]``: the binding sits mid-arm."""
    block = ConditionalBlock(label)
    body = ControlFlowRegion(f'{label}_body')
    first = body.add_state(f'{label}_zero', is_start_block=True)
    add_write(first, out, 'y = 0.0')
    second = body.add_state(f'{label}_read')
    add_write(second, out, 'y = x', 'z[k]')
    body.add_edge(first, second, dace.InterstateEdge(assignments={'k': binding}))
    block.add_branch(CodeBlock(condition), body)
    return block


def two_guarded_reads(second_binding: str = 'idx[0]', write_idx: bool = False) -> dace.SDFG:
    """``if p > 0: {if q > 0: {out[0] = z[idx[0]]}}; if q > 0: out[1] = z[<second_binding>]``."""
    sdfg = dace.SDFG('two_guarded_reads')
    sdfg.add_array('idx', [3], dace.int64)
    sdfg.add_array('z', [3], dace.float64)
    sdfg.add_array('out', [2], dace.float64)
    for name in ('p', 'q', 'k'):
        sdfg.add_symbol(name, dace.int64)
    entry = sdfg.add_state('entry', is_start_block=True)
    if write_idx:
        add_write(entry, 'idx[2]', 'y = 1')
    outer = ConditionalBlock('outer')
    outer_body = ControlFlowRegion('outer_body')
    start = outer_body.add_state('outer_start', is_start_block=True)
    inner = guarded_arm('inner', 'q > 0', 'out[0]', 'idx[0]')
    outer_body.add_node(inner)
    outer_body.add_edge(start, inner, dace.InterstateEdge())
    outer.add_branch(CodeBlock('p > 0'), outer_body)
    second = guarded_arm('second', 'q > 0', 'out[1]', second_binding)
    done = sdfg.add_state('done')
    for block in (outer, second):
        sdfg.add_node(block)
    sdfg.add_edge(entry, outer, dace.InterstateEdge())
    sdfg.add_edge(outer, second, dace.InterstateEdge())
    sdfg.add_edge(second, done, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def bindings_under_a_guard(sdfg: dace.SDFG) -> list:
    """Every edge binding ``k`` that some ``ConditionalBlock`` still encloses."""
    guarded = []
    for edge, region in ((e, r) for r in sdfg.all_control_flow_regions() for e in r.edges()):
        if 'k' not in edge.data.assignments:
            continue
        while region is not None and region is not sdfg:
            if isinstance(region, ConditionalBlock):
                guarded.append(edge.data.assignments['k'])
                break
            region = region.parent_graph
    return guarded


def run(sdfg: dace.SDFG, p: int, q: int) -> np.ndarray:
    out = np.full(2, -1.0)
    sdfg(idx=np.array([2, 0, 1], dtype=np.int64), z=np.array([10.0, 20.0, 30.0]), out=out, p=p, q=q)
    return out


def test_a_binding_constant_over_the_sdfg_leaves_every_guard():
    sdfg = two_guarded_reads()
    BranchNormalization().apply_pass(sdfg, {})
    sdfg.validate()
    assert bindings_under_a_guard(sdfg) == [], bindings_under_a_guard(sdfg)
    np.testing.assert_array_equal(run(sdfg, 1, 1), [30.0, 30.0])
    np.testing.assert_array_equal(run(sdfg, 0, 1), [-1.0, 30.0])
    np.testing.assert_array_equal(run(sdfg, 1, 0), [-1.0, -1.0])


def test_a_symbol_bound_to_two_values_keeps_its_bindings_under_their_guards():
    """Hoisting either binding would hand the other arm's read the wrong element."""
    sdfg = two_guarded_reads(second_binding='idx[1]')
    BranchNormalization().apply_pass(sdfg, {})
    assert sorted(bindings_under_a_guard(sdfg)) == ['idx[0]', 'idx[1]'], bindings_under_a_guard(sdfg)
    np.testing.assert_array_equal(run(sdfg, 1, 1), [30.0, 10.0])


def test_a_binding_that_reads_a_written_container_stays_under_its_guard():
    """The value then depends on when it is read, so moving the binding earlier can change it."""
    sdfg = two_guarded_reads(write_idx=True)
    BranchNormalization().apply_pass(sdfg, {})
    assert bindings_under_a_guard(sdfg) == ['idx[0]', 'idx[0]'], bindings_under_a_guard(sdfg)
