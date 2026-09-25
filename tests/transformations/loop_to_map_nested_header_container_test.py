# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` must route every data container a loop-body HEADER reads, at any depth.

``apply`` outlines the loop body into a NestedSDFG, and a container named by a ``LoopRegion`` or
``ConditionalBlock`` header (CloudSC's ``if plude > yrecldp_rlmin``, where the threshold is a
``float64`` scalar container, not a symbol) has to become an input connector. The header walk used
to recurse only through ``LoopRegion`` / ``ConditionalBlock`` nodes, so it stopped at a
``ConditionalBlock``'s branch -- a plain ``ControlFlowRegion``. A header below that point kept its
container as a free symbol of the body, and ``add_nested_sdfg`` types an unknown free symbol
``int``: the ``float64`` threshold reached the branch truncated to ``0`` and flipped it.

This is the defect that made CloudSC's canonicalize pipeline diverge at stage ``parallelize``; the
kernel here is the same shape stripped to the two nested headers that expose it.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import CodeBlock, ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap

N = dace.symbol('N')

#: Threshold and the value written when the guard holds. ``0 < THRESHOLD < 1`` is what makes the
#: bug observable: truncated to an ``int`` the threshold is 0, so ``thr > 0.5`` flips to False.
THRESHOLD = 0.75


def build_sdfg() -> dace.SDFG:
    """``for i: if N > 0: if thr > 0.5: B[i] = 1 else: B[i] = -1`` with ``thr`` a float64 scalar container.

    The inner ``ConditionalBlock`` sits inside the outer one's branch region, which is exactly the
    depth the old header walk could not reach.
    """
    sdfg = dace.SDFG('loop_to_map_nested_header_container')
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_scalar('thr', dace.float64)

    loop = LoopRegion('outer_loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)

    outer_if = ConditionalBlock('outer_if', sdfg=sdfg)
    loop.add_node(outer_if, is_start_block=True)
    outer_body = ControlFlowRegion('outer_body', sdfg=sdfg, parent=outer_if)
    outer_if.add_branch(CodeBlock('N > 0'), outer_body)

    inner_if = ConditionalBlock('inner_if', sdfg=sdfg)
    outer_body.add_node(inner_if, is_start_block=True)
    for label, value, condition in (('then_body', 1.0, CodeBlock('thr > 0.5')), ('else_body', -1.0, None)):
        branch = ControlFlowRegion(label, sdfg=sdfg, parent=inner_if)
        state = branch.add_state(f'{label}_state', is_start_block=True)
        tasklet = state.add_tasklet(label, {}, {'__out'}, f'__out = {value}')
        state.add_edge(tasklet, '__out', state.add_write('B'), None, dace.Memlet('B[i]'))
        inner_if.add_branch(condition, branch)

    sdfg.validate()
    return sdfg


def test_header_container_below_a_branch_is_routed():
    """The threshold reaches the lifted body as a connector, not as a re-typed ``int`` symbol."""
    sdfg = build_sdfg()
    assert sdfg.apply_transformations_repeated(LoopToMap) == 1
    sdfg.validate()

    body = next(sd for sd in sdfg.all_sdfgs_recursive() if sd is not sdfg)
    assert 'thr' not in body.symbols, f'thr was re-typed as a symbol: {body.symbols.get("thr")}'
    nsdfg_node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    assert 'thr' in nsdfg_node.in_connectors


def test_header_container_below_a_branch_keeps_its_value():
    """End-to-end: an ``int``-truncated threshold would take the else branch and write -1."""
    sdfg = build_sdfg()
    sdfg.apply_transformations_repeated(LoopToMap)

    out = np.zeros(8)
    sdfg(B=out, thr=THRESHOLD, N=8)
    assert np.array_equal(out, np.ones(8))


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__]))
