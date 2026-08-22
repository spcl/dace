# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A validation error must be printable AND must name the element that actually failed.

``state_id`` / ``node_id`` / ``edge_id`` index the element's PARENT control flow region. Resolving
them against the top-level SDFG names whichever unrelated block sits at the same index -- sending
whoever reads the message to the wrong state -- or misses entirely, and a ``__str__`` that raises
prints as ``<exception str() failed>``, hiding the very bug the error was reporting. The errors
therefore carry the region (``cfg``) that the ids belong to.
"""
from typing import Callable, Tuple

import dace
import pytest

from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg.validation import (InvalidSDFGEdgeError, InvalidSDFGError, InvalidSDFGInterstateEdgeError,
                                  InvalidSDFGNodeError)


def one_state_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('printable_error')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_state('only', is_start_block=True)
    return sdfg


@pytest.mark.parametrize('error', [
    lambda sdfg: InvalidSDFGError('boom', sdfg, 99),
    lambda sdfg: InvalidSDFGNodeError('boom', sdfg, 99, 7),
    lambda sdfg: InvalidSDFGNodeError('boom', sdfg, 0, 7),
    lambda sdfg: InvalidSDFGEdgeError('boom', sdfg, 99, 7),
    lambda sdfg: InvalidSDFGEdgeError('boom', sdfg, 0, 7),
    lambda sdfg: InvalidSDFGInterstateEdgeError('boom', sdfg, 7),
])
def test_unresolvable_ids_still_format(error):
    """Every validation error formats out-of-range ids instead of raising while formatting."""
    sdfg = one_state_sdfg()
    text = str(error(sdfg))
    assert 'boom' in text
    assert 'unresolved' in text


def test_isolated_node_in_nested_region_is_printable():
    """End-to-end: an isolated node inside a LoopRegion body raises a *readable* error.

    The state's ``state_id`` is its index in the loop body, which does not address anything in the
    top-level SDFG -- exactly the shape that used to print ``<exception str() failed>``.
    """
    sdfg = dace.SDFG('isolated_in_loop')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_transient('tmp', [1], dace.float64)

    loop = LoopRegion('loop', 'i < 4', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state('first', is_start_block=True)
    second = loop.add_state('second')
    loop.add_edge(first, second, dace.InterstateEdge())

    # An access node with no edges at all: "Isolated node".
    second.add_access('tmp')

    with pytest.raises(InvalidSDFGNodeError) as info:
        sdfg.validate()

    text = str(info.value)
    assert 'Isolated node' in text
    # The ids belong to the loop body, so that is where they must be resolved.
    assert 'at state second' in text, text
    assert 'node tmp' in text, text
    assert 'unresolved' not in text, text


########################################
# Nested failures, each with a DECOY block at the same index of the top-level SDFG


def isolated_node_in_conditional() -> dace.SDFG:
    """Failing state is index 0 of a ConditionalBlock branch; index 0 of the SDFG is ``decoy_0``."""
    sdfg = dace.SDFG('isolated_in_conditional')
    sdfg.add_transient('tmp', [1], dace.float64)

    decoy = sdfg.add_state('decoy_0', is_start_block=True)
    cond = ConditionalBlock('cblk', sdfg=sdfg)
    sdfg.add_node(cond)
    sdfg.add_edge(decoy, cond, dace.InterstateEdge())

    branch = ControlFlowRegion('branch_body', sdfg=sdfg)
    cond.add_branch(CodeBlock('1 > 0'), branch)
    failing = branch.add_state('failing_block', is_start_block=True)
    other = branch.add_state('other_block')
    branch.add_edge(failing, other, dace.InterstateEdge())

    failing.add_access('tmp')  # No edges at all -> "Isolated node".
    return sdfg


def isolated_node_in_loop() -> dace.SDFG:
    """Failing state is index 0 of a LoopRegion body; index 0 of the SDFG is ``decoy_0``."""
    sdfg = dace.SDFG('isolated_in_loop_region')
    sdfg.add_transient('tmp', [1], dace.float64)

    decoy = sdfg.add_state('decoy_0', is_start_block=True)
    loop = LoopRegion('loop', 'i < 4', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(decoy, loop, dace.InterstateEdge())

    failing = loop.add_state('failing_block', is_start_block=True)
    other = loop.add_state('other_block')
    loop.add_edge(failing, other, dace.InterstateEdge())

    failing.add_access('tmp')
    return sdfg


def isolated_node_two_levels_deep() -> dace.SDFG:
    """Failing state sits in a ConditionalBlock branch inside a LoopRegion inside the SDFG."""
    sdfg = dace.SDFG('isolated_two_levels_deep')
    sdfg.add_transient('tmp', [1], dace.float64)

    decoy = sdfg.add_state('decoy_0', is_start_block=True)
    loop = LoopRegion('loop', 'i < 4', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(decoy, loop, dace.InterstateEdge())

    inner_decoy = loop.add_state('decoy_1', is_start_block=True)
    cond = ConditionalBlock('cblk', sdfg=sdfg)
    loop.add_node(cond)
    loop.add_edge(inner_decoy, cond, dace.InterstateEdge())

    branch = ControlFlowRegion('branch_body', sdfg=sdfg)
    cond.add_branch(CodeBlock('1 > 0'), branch)
    failing = branch.add_state('failing_block', is_start_block=True)
    failing.add_access('tmp')
    return sdfg


def isolated_node_at_top_level() -> dace.SDFG:
    """Failing state is index 2 of the SDFG itself -- the case that already worked."""
    sdfg = dace.SDFG('isolated_at_top_level')
    sdfg.add_transient('tmp', [1], dace.float64)

    decoy_0 = sdfg.add_state('decoy_0', is_start_block=True)
    decoy_1 = sdfg.add_state('decoy_1')
    failing = sdfg.add_state('failing_block')
    sdfg.add_edge(decoy_0, decoy_1, dace.InterstateEdge())
    sdfg.add_edge(decoy_1, failing, dace.InterstateEdge())

    failing.add_access('tmp')
    return sdfg


NESTED_NODE_ERRORS: Tuple[Tuple[str, Callable[[], dace.SDFG]], ...] = (
    ('conditional', isolated_node_in_conditional),
    ('loop', isolated_node_in_loop),
    ('two_levels_deep', isolated_node_two_levels_deep),
    ('top_level', isolated_node_at_top_level),
)
NESTED_NODE_ERROR_IDS: Tuple[str, ...] = tuple(name for name, _ in NESTED_NODE_ERRORS)


@pytest.mark.parametrize('name,builder', NESTED_NODE_ERRORS, ids=NESTED_NODE_ERROR_IDS)
def test_node_error_names_the_failing_state(name: str, builder: Callable[[], dace.SDFG]):
    """The reported state is the one that failed, not the block at the same index of the SDFG."""
    sdfg = builder()
    with pytest.raises(InvalidSDFGNodeError) as info:
        sdfg.validate()

    text = str(info.value)
    assert 'Isolated node' in text
    assert 'at state failing_block' in text, text
    # Resolving against the SDFG named a top-level decoy, or failed to find the node inside it.
    assert 'decoy' not in text, text
    assert 'unresolved' not in text, text
    assert 'node tmp' in text, text


@pytest.mark.parametrize('name,builder', NESTED_NODE_ERRORS, ids=NESTED_NODE_ERROR_IDS)
def test_node_error_json_points_at_the_containing_region(name: str, builder: Callable[[], dace.SDFG]):
    """``(cfg_id, state_id)`` must address the failing block: consumers index the cfg list with it."""
    sdfg = builder()
    with pytest.raises(InvalidSDFGNodeError) as info:
        sdfg.validate()

    payload = info.value.to_json()
    block = sdfg.cfg_list[payload['cfg_id']].node(payload['state_id'])
    assert isinstance(block, SDFGState)
    assert block.label == 'failing_block'
    assert block.node(payload['node_id']).data == 'tmp'


def test_dataflow_edge_error_in_nested_region_names_the_failing_state():
    """``InvalidSDFGEdgeError`` resolves its state against the containing region as well."""
    sdfg = dace.SDFG('bad_edge_in_loop')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('B', [4], dace.float64)

    decoy = sdfg.add_state('decoy_0', is_start_block=True)
    loop = LoopRegion('loop', 'i < 4', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(decoy, loop, dace.InterstateEdge())

    failing = loop.add_state('failing_block', is_start_block=True)
    read = failing.add_access('A')
    write = failing.add_access('B')
    # Reads eight elements out of a four-element array.
    failing.add_edge(read, None, write, None, dace.Memlet('A[0:8] -> [0:8]'))

    with pytest.raises(InvalidSDFGEdgeError) as info:
        sdfg.validate()

    text = str(info.value)
    assert 'at state failing_block' in text, text
    assert 'decoy' not in text, text
    assert 'unresolved' not in text, text


def test_interstate_edge_error_in_nested_region_names_the_failing_edge():
    """``InvalidSDFGInterstateEdgeError`` resolves ``edge_id`` against the containing region."""
    sdfg = dace.SDFG('bad_isedge_in_region')
    sdfg.add_array('A', [4], dace.float64)

    decoy = sdfg.add_state('decoy_0', is_start_block=True)
    region = ControlFlowRegion('region', sdfg=sdfg)
    sdfg.add_node(region)
    sdfg.add_edge(decoy, region, dace.InterstateEdge())

    src = region.add_state('edge_src', is_start_block=True)
    dst = region.add_state('edge_dst')
    region.add_edge(src, dst, dace.InterstateEdge(assignments={'k': 'never_defined_symbol'}))

    with pytest.raises(InvalidSDFGInterstateEdgeError) as info:
        sdfg.validate()

    text = str(info.value)
    # Region edge 0 is edge_src -> edge_dst; SDFG edge 0 is decoy_0 -> region.
    assert 'at edge edge_src -> edge_dst' in text, text
    assert 'decoy' not in text, text
    assert 'unresolved' not in text, text

    payload = info.value.to_json()
    edge = sdfg.cfg_list[payload['cfg_id']].edges()[payload['isedge_id']]
    assert (edge.src.label, edge.dst.label) == ('edge_src', 'edge_dst')


def test_sdfg_level_error_still_formats():
    """Errors with no ``state_id`` (SDFG-level checks) are unaffected."""
    sdfg = dace.SDFG('empty_sdfg')
    with pytest.raises(InvalidSDFGError) as info:
        sdfg.validate()
    assert 'at least one state' in str(info.value)


def test_debuginfo_is_reported_for_every_node_type_that_carries_it():
    """``_getlineinfo`` is a static type check now; map entries must keep their line info.

    ``MapEntry``/``ConsumeEntry`` inherit ``debuginfo`` indirectly from their ``Map``/``Consume``,
    so a type check that only knew about tasklets and access nodes would silently drop it.
    """
    sdfg = dace.SDFG('lineinfo')
    sdfg.add_array('A', [4], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    entry, _ = state.add_map('m', dict(i='0:4'))
    entry.map.debuginfo = dace.dtypes.DebugInfo(42, 3, 42, 10, 'somewhere.py')
    tasklet = state.add_tasklet('t', {}, {'o'}, 'o = 1')
    tasklet.debuginfo = dace.dtypes.DebugInfo(7, 0, 7, 0, 'elsewhere.py')

    entry_error = str(InvalidSDFGNodeError('boom', sdfg, 0, state.node_id(entry)))
    assert 'somewhere.py' in entry_error and 'line 42' in entry_error and 'column 3' in entry_error

    tasklet_error = str(InvalidSDFGNodeError('boom', sdfg, 0, state.node_id(tasklet)))
    assert 'elsewhere.py' in tasklet_error and 'line 7' in tasklet_error
    assert 'column' not in tasklet_error

    # A state carries no debuginfo -- it must contribute no source location, and must not raise.
    assert 'Originating' not in str(InvalidSDFGNodeError('boom', sdfg, 0, None))


########################################
# Acceptance: the fix changes only what the message says, never which SDFGs validate


def valid_conditional() -> dace.SDFG:
    sdfg = dace.SDFG('valid_conditional')
    sdfg.add_array('A', [4], dace.float64)
    pre = sdfg.add_state('pre', is_start_block=True)
    cond = ConditionalBlock('cblk', sdfg=sdfg)
    sdfg.add_node(cond)
    sdfg.add_edge(pre, cond, dace.InterstateEdge())
    branch = ControlFlowRegion('branch_body', sdfg=sdfg)
    cond.add_branch(CodeBlock('1 > 0'), branch)
    body = branch.add_state('body', is_start_block=True)
    body.add_edge(body.add_tasklet('t', {}, {'o'}, 'o = 1'), 'o', body.add_access('A'), None, dace.Memlet('A[0]'))
    return sdfg


def valid_loop() -> dace.SDFG:
    sdfg = dace.SDFG('valid_loop')
    sdfg.add_array('A', [4], dace.float64)
    pre = sdfg.add_state('pre', is_start_block=True)
    loop = LoopRegion('loop', 'i < 4', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, dace.InterstateEdge())
    body = loop.add_state('body', is_start_block=True)
    body.add_edge(body.add_tasklet('t', {}, {'o'}, 'o = 1'), 'o', body.add_access('A'), None, dace.Memlet('A[i]'))
    return sdfg


@pytest.mark.parametrize('builder', [valid_conditional, valid_loop], ids=['conditional', 'loop'])
def test_valid_nested_regions_still_validate(builder: Callable[[], dace.SDFG]):
    """Acceptance is unchanged: graphs that validated before still validate."""
    builder().validate()


@pytest.mark.parametrize('name,builder', NESTED_NODE_ERRORS, ids=NESTED_NODE_ERROR_IDS)
def test_invalid_sdfgs_are_still_rejected(name: str, builder: Callable[[], dace.SDFG]):
    """Acceptance is unchanged in the other direction, too."""
    with pytest.raises(InvalidSDFGNodeError):
        builder().validate()


def test_unserializable_invalid_sdfg_still_reports():
    """``validate_sdfg`` dumps the invalid graph for inspection. That dump walks the very graph that
    is broken, so it can raise on its own -- and an exception from the dump would replace the
    diagnostic with an unrelated traceback."""
    sdfg = dace.SDFG('unserializable_invalid')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('B', [4], dace.float64)

    inner = dace.SDFG('inner')
    inner.add_array('A', [4], dace.float64)
    inner.add_array('B', [4], dace.float64)
    istate = inner.add_state('istate', is_start_block=True)
    istate.add_nedge(istate.add_read('A'), istate.add_write('B'), dace.Memlet('A[0:4]'))

    state = sdfg.add_state('outer', is_start_block=True)
    nsdfg = state.add_nested_sdfg(inner, {'A': None}, {'B': None})
    state.add_edge(state.add_read('A'), None, nsdfg, 'A', dace.Memlet('A[0:4]'))
    state.add_edge(nsdfg, 'B', state.add_write('B'), None, dace.Memlet('B[0:4]'))
    sdfg.validate()

    # An owner-less block: invalid, and ``SDFGState.to_json`` dereferences ``self.sdfg`` unguarded.
    istate.sdfg = None

    with pytest.raises(InvalidSDFGNodeError) as info:
        sdfg.validate()
    assert 'Node validation failed' in str(info.value)


if __name__ == '__main__':
    test_unresolvable_ids_still_format(lambda sdfg: InvalidSDFGNodeError('boom', sdfg, 99, 7))
    test_isolated_node_in_nested_region_is_printable()
    for _name, _builder in NESTED_NODE_ERRORS:
        test_node_error_names_the_failing_state(_name, _builder)
        test_node_error_json_points_at_the_containing_region(_name, _builder)
    test_dataflow_edge_error_in_nested_region_names_the_failing_state()
    test_interstate_edge_error_in_nested_region_names_the_failing_edge()
    test_sdfg_level_error_still_formats()
    test_debuginfo_is_reported_for_every_node_type_that_carries_it()
    test_valid_nested_regions_still_validate(valid_conditional)
    test_valid_nested_regions_still_validate(valid_loop)
    for _name, _builder in NESTED_NODE_ERRORS:
        test_invalid_sdfgs_are_still_rejected(_name, _builder)

    test_unserializable_invalid_sdfg_still_reports()
