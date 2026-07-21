# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A validation error must always be printable.

``state_id`` / ``node_id`` / ``edge_id`` index the element's PARENT control flow region, but the
``__str__`` implementations resolve them against ``self.sdfg``. For a state nested in a LoopRegion or
ConditionalBlock those are different graphs, so the lookup misses -- and a ``__str__`` that raises
prints as ``<exception str() failed>``, hiding the very bug the error was reporting.
"""
import dace
import pytest

from dace.sdfg.state import LoopRegion
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
    # The ids do not resolve against the top-level SDFG; the message must say so, not raise.
    assert 'unresolved' in text


if __name__ == '__main__':
    test_unresolvable_ids_still_format(lambda sdfg: InvalidSDFGNodeError('boom', sdfg, 99, 7))
    test_isolated_node_in_nested_region_is_printable()
