# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for ``'ai'`` as a reserved library node implementation name. """

import os
import sys

import numpy as np
import pytest

import dace
import dace.library
import dace.libraries.ai as ai
from dace import nodes
from dace.transformation import transformation as xf

sys.path.insert(0, os.path.dirname(__file__))
from ai_test_utils import stub_provider  # noqa: E402

from dace.libraries.ai.backend import TaskletSpec  # noqa: E402


@dace.library.node
class DoublerNode(nodes.LibraryNode):
    """ A library node declared here, so that it is never seen by DaCe at import time. """

    implementations = {}
    default_implementation = 'ai'

    def __init__(self, name='doubler', **kwargs):
        super().__init__(name, inputs={'_inp'}, outputs={'_out'}, **kwargs)


class UndecoratedNode(DoublerNode):
    """ A subclass that never passes through ``@dace.library.node``. """

    implementations = {}
    default_implementation = 'ai'


def _doubler_sdfg() -> dace.SDFG:
    """
    Builds a one-element SDFG around a :class:`DoublerNode`.

    :return: The SDFG.
    """
    sdfg = dace.SDFG('ai_doubler')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    state = sdfg.add_state()
    node = DoublerNode()
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_inp', dace.Memlet('A[0]'))
    state.add_edge(node, '_out', state.add_write('B'), None, dace.Memlet('B[0]'))
    return sdfg


def test_ai_is_available_on_every_library_node():
    from dace.libraries.blas.nodes.gemm import Gemm

    # A node shipped with DaCe, one declared in this test, and an undecorated subclass
    assert 'ai' in Gemm.available_implementations()
    assert 'ai' in DoublerNode.available_implementations()
    assert 'ai' in UndecoratedNode.available_implementations()

    # ... without being registered into the implementations dictionary
    assert 'ai' not in Gemm.implementations
    assert 'ai' not in DoublerNode.implementations

    # The registered implementations are still reported
    assert 'pure' in Gemm.available_implementations()


def test_expansion_class_is_bound_and_cached():
    from dace.libraries.blas.nodes.gemm import Gemm

    gemm_expansion = ai.ExpandAI.for_node_class(Gemm)
    doubler_expansion = ai.ExpandAI.for_node_class(DoublerNode)

    assert gemm_expansion._match_node is Gemm
    assert doubler_expansion._match_node is DoublerNode
    assert gemm_expansion is not doubler_expansion
    # Cached, so repeated expansions of the same node type reuse one class
    assert ai.ExpandAI.for_node_class(Gemm) is gemm_expansion


def test_registering_ai_raises():
    with pytest.raises(ValueError, match='reserved'):

        @dace.library.register_expansion(DoublerNode, 'ai')
        class ExpandForeignAI(xf.ExpandTransformation):
            environments = []

            @staticmethod
            def expansion(node, state, sdfg, **kwargs):
                return nodes.Tasklet('nothing', code='pass')

    class ExpandOther(xf.ExpandTransformation):
        environments = []

    with pytest.raises(ValueError, match='reserved'):
        DoublerNode.register_implementation('ai', ExpandOther)

    assert 'ai' not in DoublerNode.implementations


def test_unknown_implementation_lists_ai():
    sdfg = _doubler_sdfg()
    state = sdfg.states()[0]
    node = next(n for n in state.nodes() if isinstance(n, DoublerNode))

    with pytest.raises(KeyError, match='ai'):
        state.expand_library_node(node, 'no_such_implementation')


def test_expand_with_ai_produces_a_working_tasklet():
    sdfg = _doubler_sdfg()
    state = sdfg.states()[0]
    node = next(n for n in state.nodes() if isinstance(n, DoublerNode))

    spec = TaskletSpec(code='_out = 2.0 * _inp;', language='CPP', notes='doubling')
    with stub_provider(spec) as provider:
        assert node.expand(state, 'ai') == 'ai'

    assert provider.calls, 'the provider was never asked to generate code'
    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert tasklet.code.as_string.strip() == '_out = 2.0 * _inp;'
    assert tasklet.language == dace.Language.CPP

    a = np.array([21.0], dtype=np.float64)
    b = np.zeros([1], dtype=np.float64)
    sdfg(A=a, B=b)
    assert np.allclose(b, 42.0)


def test_expand_via_state_helper():
    sdfg = _doubler_sdfg()
    state = sdfg.states()[0]
    node = next(n for n in state.nodes() if isinstance(n, DoublerNode))

    with stub_provider(TaskletSpec(code='_out = _inp;')):
        assert state.expand_library_node(node, 'ai') == 'ai'
    assert any(isinstance(n, nodes.Tasklet) for n in state.nodes())


def test_default_implementation_selects_ai():
    sdfg = _doubler_sdfg()
    state = sdfg.states()[0]
    node = next(n for n in state.nodes() if isinstance(n, DoublerNode))

    # No explicit implementation: falls through to DoublerNode.default_implementation
    with stub_provider(TaskletSpec(code='_out = _inp;')):
        assert node.expand(state) == 'ai'


def test_registered_implementations_still_work():
    from dace.libraries.blas.nodes.gemm import Gemm

    sdfg = dace.SDFG('ai_gemm_pure')
    for name in ('A', 'B'):
        sdfg.add_array(name, [8, 8], dace.float64)
    sdfg.add_array('C', [8, 8], dace.float64)
    state = sdfg.add_state()
    gemm = Gemm('gemm')
    state.add_node(gemm)
    state.add_edge(state.add_read('A'), None, gemm, '_a', dace.Memlet('A[0:8, 0:8]'))
    state.add_edge(state.add_read('B'), None, gemm, '_b', dace.Memlet('B[0:8, 0:8]'))
    state.add_edge(gemm, '_c', state.add_write('C'), None, dace.Memlet('C[0:8, 0:8]'))

    assert gemm.expand(state, 'pure') == 'pure'

    rng = np.random.default_rng(0)
    a, b = rng.random((8, 8)), rng.random((8, 8))
    c = np.zeros((8, 8))
    sdfg(A=a, B=b, C=c)
    assert np.allclose(c, a @ b)


if __name__ == '__main__':
    test_ai_is_available_on_every_library_node()
    test_expansion_class_is_bound_and_cached()
    test_registering_ai_raises()
    test_unknown_implementation_lists_ai()
    test_expand_with_ai_produces_a_working_tasklet()
    test_expand_via_state_helper()
    test_default_implementation_selects_ai()
    test_registered_implementations_still_work()
