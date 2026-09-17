# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Live test of the reserved ``'ai'`` implementation on a library node that knows nothing about it.

``tests/library/addlib`` is an ordinary external DaCe library with a single ``pure`` expansion. It
gains ``'ai'`` without being modified, registered, or re-imported.
"""

import os
import sys

import numpy as np
import pytest

import dace
from dace import nodes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import addlib  # noqa: E402

DESCRIPTION_HINT = 'Add one to the input value and write it to the output.'


@pytest.mark.ai
def test_external_library_node_expands_with_ai():
    # The node's own library only ships a 'pure' implementation
    assert 'ai' not in addlib.AddNode.implementations
    assert 'ai' in addlib.AddNode.available_implementations()

    sdfg = dace.SDFG('ai_addlib')
    sdfg.add_array('A', [1], dace.float32)
    sdfg.add_array('B', [1], dace.float32)
    state = sdfg.add_state()
    node = addlib.AddNode('add')
    # The node carries no description, so the model works from its name and connectors alone
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet('A[0]'))
    state.add_edge(node, '_b', state.add_write('B'), None, dace.Memlet('B[0]'))

    with dace.config.set_temporary('ai', 'extra_instructions', value=DESCRIPTION_HINT):
        assert node.expand(state, 'ai') == 'ai'

    assert any(isinstance(n, nodes.Tasklet) for n in state.nodes())
    assert not any(isinstance(n, addlib.AddNode) for n in state.nodes())

    a = np.array([41.0], dtype=np.float32)
    b = np.zeros([1], dtype=np.float32)
    sdfg(A=a, B=b)
    assert np.allclose(b, 42.0)


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'ai'])
