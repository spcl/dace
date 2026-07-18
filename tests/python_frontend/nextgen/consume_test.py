# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for explicit consume scopes (``@dace.consume``/``@dace.consumescope``)
in the next-generation Python frontend: decorated functions lower to
:class:`~dace.sdfg.analysis.schedule_tree.treenodes.ConsumeScope` nodes with
real :class:`~dace.sdfg.nodes.ConsumeEntry` metadata, streams register as
:class:`~dace.data.Stream` containers (``dace.define_stream``), and the
popped element enters the body as a dynamic stream read.

``tree_to_sdfg`` does not lower ``ConsumeScope`` yet: consume programs build
correct schedule trees but converting them to SDFGs raises
NotImplementedError (an explicit backend gap, like ViewNode/RefSetNode and
callback nodes before their lowering rounds).
"""
import pytest

import dace
from dace import data, nodes
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


@dace.program
def fibonacci(iv: dace.int32[1], res: dace.float32[1]):
    S = dace.define_stream(dace.int32, 0)
    with dace.tasklet:
        i << iv
        s >> S
        s = i

    @dace.consume(S, 4)
    def scope(elem, p):
        sout >> S(-1)
        val >> res(-1, lambda a, b: a + b)
        if elem == 1:
            val = 1
        elif elem > 1:
            sout = elem - 1
            sout = elem - 2


def test_consume_tasklet_form():
    tree = nextgen.parse_program(fibonacci)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    # The stream registered as a real Stream container.
    assert isinstance(tree.containers['S'], data.Stream)

    scopes = _nodes_of_type(tree, tn.ConsumeScope)
    assert len(scopes) == 1
    entry = scopes[0].node
    assert isinstance(entry, nodes.ConsumeEntry)
    assert entry.consume.pe_index == 'p'
    assert str(entry.consume.num_pes) == '4'
    assert entry.consume.condition is None  # Run until the stream is empty

    # The popped element is a dynamic stream read into the tasklet; the
    # stream push-back and WCR result are dynamic outputs.
    tasklets = [node for node in scopes[0].preorder_traversal() if isinstance(node, tn.TaskletNode)]
    assert len(tasklets) == 1
    tasklet = tasklets[0]
    assert tasklet.in_memlets['elem'].data == 'S'
    assert tasklet.in_memlets['elem'].dynamic
    assert tasklet.out_memlets['sout'].data == 'S'
    assert tasklet.out_memlets['val'].data == 'res'
    assert tasklet.out_memlets['val'].wcr is not None


def test_consumescope_form():
    N = dace.symbol('N')

    @dace.program
    def consumer(iv: dace.int32[1], out: dace.float32[4]):
        S = dace.define_stream(dace.int32, 0)
        with dace.tasklet:
            i << iv
            s >> S
            s = i

        @dace.consumescope(S, 4)
        def scope(elem, p):
            out[p] = elem * 2.0

    tree = nextgen.parse_program(consumer)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    scopes = _nodes_of_type(tree, tn.ConsumeScope)
    assert len(scopes) == 1
    # The element materializes through a leading pop tasklet reading the
    # stream dynamically, then the (canonicalized) body statements follow.
    children = scopes[0].children
    assert isinstance(children[0], tn.TaskletNode)
    assert children[0].in_memlets['__stream'].data == 'S'
    assert children[0].in_memlets['__stream'].dynamic
    assert len(children) > 1


def test_consume_condition_and_chunksize():

    @dace.program
    def conditional_consume(iv: dace.int32[1], res: dace.float32[1]):
        S = dace.define_stream(dace.int32, 0)
        with dace.tasklet:
            i << iv
            s >> S
            s = i

        @dace.consume(S, 2, lambda: True, 3)
        def scope(elem, p):
            val >> res(-1, lambda a, b: a + b)
            val = elem

    tree = nextgen.parse_program(conditional_consume)
    scopes = _nodes_of_type(tree, tn.ConsumeScope)
    assert len(scopes) == 1
    consume = scopes[0].node.consume
    assert consume.condition is not None
    assert consume.chunksize == 3


def test_consume_malformed_fallback():

    @dace.program
    def missing_pes(iv: dace.int32[1], res: dace.float32[1]):
        S = dace.define_stream(dace.int32, 0)
        with dace.tasklet:
            i << iv
            s >> S
            s = i

        @dace.consume(S)
        def scope(elem, p):
            val >> res(-1, lambda a, b: a + b)
            val = elem

    # Fewer than two decorator arguments violates the classic contract; the
    # decorated def stays unrecognized and falls back to a callback.
    tree = nextgen.parse_program(missing_pes)
    assert len(_nodes_of_type(tree, tn.PythonCallbackNode)) >= 1
    assert not _nodes_of_type(tree, tn.ConsumeScope)


def test_consume_to_sdfg_gap():
    tree = nextgen.parse_program(fibonacci)
    with pytest.raises(NotImplementedError):
        tree.as_sdfg()


if __name__ == '__main__':
    test_consume_tasklet_form()
    test_consumescope_form()
    test_consume_condition_and_chunksize()
    test_consume_malformed_fallback()
    test_consume_to_sdfg_gap()
