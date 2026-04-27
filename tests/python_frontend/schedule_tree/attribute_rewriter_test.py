# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast
import pytest

import dace
import numpy as np
from dace.frontend.python import astutils
from dace.frontend.python.schedule_tree import AttributeRewriter
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _rewrite_expression(source: str, context):
    rewriter = AttributeRewriter(lambda: dict(context))
    expr = ast.parse(source, mode='eval').body
    return astutils.unparse(rewriter.rewrite_expression(expr))


def _rewrite_assignment(source: str, context):
    rewriter = AttributeRewriter(lambda: dict(context))
    assign = ast.parse(source).body[0]
    rewritten = rewriter.rewrite_assignment(assign.targets[0], assign.value)
    return None if rewritten is None else astutils.unparse(rewritten)


def test_attribute_rewriter_rewrites_descriptor_loads_and_stores():

    class ArrayDescriptor:

        def __set_name__(self, owner, name):
            self.name = '_' + name

        def __get__(self, obj, objtype=None):
            return getattr(obj, self.name)

        def __set__(self, obj, value):
            setattr(obj, self.name, value)

    class DescriptorHolder:
        arr = ArrayDescriptor()

        def __init__(self):
            self.arr = None

    descriptor_holder = DescriptorHolder()
    context = {'descriptor_holder': descriptor_holder}

    assert _rewrite_assignment('descriptor_holder.arr = A',
                               context) == ("type(descriptor_holder).__dict__['arr'].__set__(descriptor_holder, A)")
    assert _rewrite_expression(
        'descriptor_holder.arr',
        context) == ("type(descriptor_holder).__dict__['arr'].__get__(descriptor_holder, type(descriptor_holder))")


def test_attribute_rewriter_rewrites_custom_getattribute_and_setattr():

    class Proxy:

        def __getattribute__(self, name):
            return object.__getattribute__(self, name)

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)

    proxy = Proxy()
    context = {'proxy': proxy}

    assert _rewrite_expression('proxy.value', context) == "type(proxy).__getattribute__(proxy, 'value')"
    assert _rewrite_assignment('proxy.value = A', context) == "type(proxy).__setattr__(proxy, 'value', A)"


def test_attribute_rewriter_preserves_plain_attribute_syntax():

    class Holder:

        def __init__(self):
            self.value = None

    holder = Holder()
    context = {'holder': holder}

    assert _rewrite_expression('holder.value', context) == 'holder.value'
    assert _rewrite_assignment('holder.value = A', context) is None


def test_attribute_rewriter_preserves_plain_method_syntax():

    class Holder:

        def method(self, value):
            return value

        @staticmethod
        def static_method(value):
            return value

        @classmethod
        def class_method(cls, value):
            return value

    holder = Holder()
    context = {'holder': holder}

    assert _rewrite_expression('holder.method', context) == 'holder.method'
    assert _rewrite_expression('holder.method(A)', context) == 'holder.method(A)'
    assert _rewrite_expression('holder.static_method(A)', context) == 'holder.static_method(A)'
    assert _rewrite_expression('holder.class_method(A)', context) == 'holder.class_method(A)'


def test_attribute_rewriter_preserves_mpi4py_method_syntax():
    MPI = pytest.importorskip('mpi4py.MPI')

    commworld = MPI.COMM_WORLD
    context = {'commworld': commworld}

    assert _rewrite_expression('commworld.Bcast(A)', context) == 'commworld.Bcast(A)'


def test_schedule_tree_lowers_plain_object_registered_methods():
    MPI = pytest.importorskip('mpi4py.MPI')

    commworld = MPI.COMM_WORLD

    @dace.program
    def comm_world_bcast(A: dace.int32[10]):
        commworld.Bcast(A)

    stree = comm_world_bcast.to_schedule_tree(np.zeros((10, ), dtype=np.int32))

    assert isinstance(stree.children[0], tn.LibraryCall)
    assert stree.children[0].node.name == 'Bcast'
    assert stree.children[0].node.properties['receiver_class'] == 'Intracomm'
    assert stree.children[0].node.properties['receiver'] == 'commworld'
    assert not any(isinstance(node, tn.StatementNode) for node in stree.preorder_traversal())


def test_schedule_tree_infers_plain_object_registered_method_results():
    MPI = pytest.importorskip('mpi4py.MPI')

    commworld = MPI.COMM_WORLD

    @dace.program
    def comm_world_isend(A: dace.int32[1]):
        req = commworld.Isend(A, 0, 0)
        return req

    stree = comm_world_isend.to_schedule_tree(np.zeros((1, ), dtype=np.int32))
    library_calls = [node for node in stree.preorder_traversal() if isinstance(node, tn.LibraryCall)]

    assert len(library_calls) == 1
    assert library_calls[0].node.name == 'Isend'
    request_name = library_calls[0].out_memlets['out'].data
    request_desc = stree.containers[request_name]
    assert isinstance(request_desc, dace.data.Array)
    assert tuple(request_desc.shape) == (1, )
    assert isinstance(request_desc.dtype, dace.dtypes.opaque)
    assert not any(isinstance(node, tn.StatementNode) for node in stree.preorder_traversal())


if __name__ == '__main__':
    pytest.main([__file__])
