# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that validation errors can be formatted for every kind of control flow block."""
import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.sdfg.validation import InvalidSDFGEdgeError, InvalidSDFGError, InvalidSDFGNodeError


def _sdfg_with_conditional_block():
    sdfg = dace.SDFG('error_message')
    start = sdfg.add_state('start', is_start_block=True)
    cond = ConditionalBlock('cond')
    sdfg.add_node(cond)
    branch = ControlFlowRegion('branch', sdfg=sdfg)
    branch.add_state('inner', is_start_block=True)
    cond.add_branch(CodeBlock('True'), branch)
    sdfg.add_edge(start, cond, dace.InterstateEdge())
    return sdfg, sdfg.node_id(cond)


def test_error_on_conditional_block():
    sdfg, block_id = _sdfg_with_conditional_block()
    assert 'at state cond' in str(InvalidSDFGError('message', sdfg, block_id))


def test_node_error_on_conditional_block():
    sdfg, block_id = _sdfg_with_conditional_block()
    assert 'at state cond' in str(InvalidSDFGNodeError('message', sdfg, block_id, 0))


def test_edge_error_on_conditional_block():
    sdfg, block_id = _sdfg_with_conditional_block()
    assert 'at state cond' in str(InvalidSDFGEdgeError('message', sdfg, block_id, 0))


if __name__ == '__main__':
    test_error_on_conditional_block()
    test_node_error_on_conditional_block()
    test_edge_error_on_conditional_block()
