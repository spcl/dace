# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the stream call surface (``stream.push(value)`` / ``stream.pop()``),
which replaces the legacy statement-level memlet-shift spelling
(``A[i] >> ostream(-1)`` / ``ostream >> oarray``) that the next-generation
frontend deliberately does not support.

Outside a dataflow scope both methods lower through the replacement registry
(:mod:`dace.frontend.python.replacements.streams`); inside one, ``push`` is
emitted directly as a tasklet with a dynamic stream write
(:mod:`dace.frontend.python.nextgen.lowering.mechanisms.streams`), because
deferred replacement expansion cannot add state machinery there.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = 64


def _nodes_of_type(tree, node_type):
    return [node for node in tree.preorder_traversal() if isinstance(node, node_type)]


def _filter_program():

    @dace.program
    def prog(A: dace.float32[N]):
        ostream = dace.define_stream(dace.float32, N)
        out = dace.define_local([N], dace.float32)
        out[:] = 0.0
        for i in dace.map[0:N]:
            if A[i] >= 0.5:
                ostream.push(A[i])
        out[:] = ostream.pop()
        return out

    return prog


def test_push_in_map_is_a_dynamic_stream_write():
    tree = nextgen.parse_program(_filter_program())
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    tasklets = [node for node in _nodes_of_type(tree, tn.TaskletNode) if 'ostream' in str(node.out_memlets)]
    assert len(tasklets) == 1
    memlet = tasklets[0].out_memlets['__out']
    assert memlet.data == 'ostream'
    assert memlet.dynamic


def test_pop_is_a_replacement_call():
    tree = nextgen.parse_program(_filter_program())
    calls = _nodes_of_type(tree, tn.ReplacementCallNode)
    assert [call.qualname for call in calls] == ['pop']
    assert calls[0].receiver == 'ostream'


def test_filter_execution():
    func = nextgen.parse_program(_filter_program()).as_sdfg().compile()
    A = np.random.rand(N).astype(np.float32)
    result = func(A=A)
    expected = A[A >= 0.5]
    # The stream drains in nondeterministic order, into the buffer the program
    # zeroed itself: a pop writes only the elements the stream held.
    assert np.allclose(np.sort(result[:expected.size]), np.sort(expected))
    assert (result[expected.size:] == 0).all()


def test_top_level_push_execution():
    """A push outside any dataflow scope goes through the registry expansion."""

    @dace.program
    def prog(A: dace.float32[N]):
        ostream = dace.define_stream(dace.float32, N)
        out = dace.define_local([N], dace.float32)
        out[:] = 0.0
        ostream.push(A[0])
        ostream.push(A[1])
        out[:] = ostream.pop()
        return out

    tree = nextgen.parse_program(prog)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    assert [call.qualname for call in _nodes_of_type(tree, tn.ReplacementCallNode)] == ['push', 'push', 'pop']

    func = tree.as_sdfg().compile()
    A = np.random.rand(N).astype(np.float32)
    result = func(A=A)
    assert np.allclose(np.sort(result[:2]), np.sort(A[:2]))
    assert (result[2:] == 0).all()


def test_pop_with_explicit_count_execution():

    @dace.program
    def prog(A: dace.float32[N]):
        ostream = dace.define_stream(dace.float32, N)
        out = dace.define_local([4], dace.float32)
        out[:] = 0.0
        ostream.push(A[0])
        out[:] = ostream.pop(4)
        return out

    tree = nextgen.parse_program(prog)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    func = tree.as_sdfg().compile()
    A = np.random.rand(N).astype(np.float32)
    result = func(A=A)
    assert result[0] == A[0]
    assert (result[1:] == 0).all()


def test_pop_into_a_named_array_drains_in_place():
    """A pop whose result the program assigns into an array IN FULL writes
    that array itself: no buffer of its own, no initialization of its own, and
    so nothing between the pushes and the drain -- the shape the code
    generator turns into a zero-copy stream-array view."""

    @dace.program
    def prog(A: dace.float32[N], out: dace.float32[N]):
        ostream = dace.define_stream(dace.float32, N)
        for i in dace.map[0:N]:
            if A[i] >= 0.5:
                ostream.push(A[i])
        out[:] = ostream.pop()

    A = np.random.rand(N).astype(np.float32)
    out = np.zeros(N, np.float32)
    tree = nextgen.parse_program(prog, A, out)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    sdfg = tree.as_sdfg()
    sdfg.simplify()
    drains = [(state, edge) for state in sdfg.states() for edge in state.edges()
              if edge.data.data == 'ostream' and getattr(edge.dst, 'data', None) == 'out']
    assert len(drains) == 1, 'the drain writes the destination itself, with nothing in between'
    assert 'dace::ArrayStreamView' in sdfg.generate_code()[0].clean_code

    out[:] = -1.0
    sdfg(A=A, out=out)
    expected = A[A >= 0.5]
    assert np.allclose(np.sort(out[:expected.size]), np.sort(expected))


def test_pop_without_a_destination_zero_pads():
    """A pop that has to produce its own buffer pads it instead: its caller
    never had the buffer in hand to initialize it."""

    @dace.program
    def prog(A: dace.float32[N]):
        ostream = dace.define_stream(dace.float32, N)
        ostream.push(A[0])
        popped = ostream.pop()
        return popped

    tree = nextgen.parse_program(prog)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    A = np.random.rand(N).astype(np.float32)
    result = tree.as_sdfg().compile()(A=A)
    assert result[0] == A[0]
    assert (result[1:] == 0).all()


if __name__ == '__main__':
    test_push_in_map_is_a_dynamic_stream_write()
    test_pop_is_a_replacement_call()
    test_filter_execution()
    test_top_level_push_execution()
    test_pop_with_explicit_count_execution()
    test_pop_into_a_named_array_drains_in_place()
    test_pop_without_a_destination_zero_pads()
