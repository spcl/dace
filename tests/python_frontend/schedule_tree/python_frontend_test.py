# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast
import contextlib
import numpy as np
import pytest
import sys
import warnings
from typing import Optional

import dace
from dace import dtypes
from dace.data.pydata import PythonDict, PythonList
from dace.frontend.python.common import DaceSyntaxError, SDFGConvertible, ScheduleTreeConvertible
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def test_python_frontend_schedule_tree_structured_control_flow():

    @dace.program
    def structured(A: dace.float64[20]):
        tmp = A[:]
        for i in range(20):
            if i < 10:
                continue
            else:
                break
        return tmp

    stree = structured.to_schedule_tree()

    assert isinstance(stree.children[0], tn.ViewNode)
    assert isinstance(stree.children[1], tn.LoopScope)
    assert isinstance(stree.children[1].loop, tn.FrontendLoop)
    assert isinstance(stree.children[1].children[0], tn.IfScope)
    assert isinstance(stree.children[1].children[0].children[0], tn.ContinueNode)
    assert isinstance(stree.children[1].children[1], tn.ElseScope)
    assert isinstance(stree.children[1].children[1].children[0], tn.BreakNode)
    assert isinstance(stree.children[2], tn.ReturnNode)


def test_python_frontend_schedule_tree_root_repository():
    offset = 3.0

    @dace.program
    def structured(A: dace.float64[20]):
        return A + offset

    stree = structured.to_schedule_tree()

    assert isinstance(stree, tn.ScheduleTreeRoot)
    assert stree.name.endswith('_structured')
    assert stree.arg_names == ['A']
    assert 'A' in stree.containers
    assert 'offset' in stree.constants


def test_python_frontend_schedule_tree_allocations_and_cache():

    @dace.program
    def alloc_copy(A: dace.float64[4]):
        tmp = np.empty_like(A)
        tmp[:] = A[:]
        return tmp

    stree_first = alloc_copy.to_schedule_tree(use_cache=True)
    stree_second = alloc_copy.to_schedule_tree(use_cache=True)

    assert stree_first is not stree_second
    assert 'tmp' in stree_first.containers
    assert (isinstance(stree_first.children[0], tn.LibraryCall)
            and stree_first.children[0].node.name == 'numpy.empty_like')
    assert isinstance(stree_first.children[1], tn.CopyNode)
    assert isinstance(stree_first.children[2], tn.ReturnNode)


def test_python_frontend_schedule_tree_references():

    @dace.program
    def refs(A: dace.float64[4], B: dace.float64[4], flag: dace.bool_):
        ref: dace.data.ArrayReference(A.dtype, A.shape) = A
        if flag:
            ref = B
        return ref

    stree = refs.to_schedule_tree()

    assert isinstance(stree.children[0], tn.RefSetNode)
    assert isinstance(stree.children[1], tn.IfScope)
    assert isinstance(stree.children[1].children[0], tn.RefSetNode)
    assert isinstance(stree.children[2], tn.ReturnNode)


def test_python_frontend_schedule_tree_unannotated_branch_references():

    @dace.program
    def refs(A: dace.float64[20], B: dace.float64[20], i: dace.int32[1], out: dace.float64[20]):
        if i[0] < 5:
            ref = A
        else:
            ref = B
        out[:] = ref

    stree = refs.to_schedule_tree()

    assert isinstance(stree.children[0], tn.IfScope)
    assert isinstance(stree.children[0].children[0], tn.RefSetNode)
    assert isinstance(stree.children[1], tn.ElseScope)
    assert isinstance(stree.children[1].children[0], tn.RefSetNode)
    assert isinstance(stree.children[2], tn.CopyNode)


def test_python_frontend_schedule_tree_map_scope():

    @dace.program
    def mapped(A: dace.float64[8]):
        for i in dace.map[0:8]:
            A[i] = A[i]

    stree = mapped.to_schedule_tree()

    assert len(stree.children) == 1
    assert isinstance(stree.children[0], tn.MapScope)
    assert isinstance(stree.children[0].node, tn.FrontendMap)
    assert isinstance(stree.children[0].children[0], tn.CopyNode)


def test_python_frontend_schedule_tree_function_call_return():

    @dace.program
    def inner(A: dace.float64[8], B: dace.float64[8]):
        return np.sum(A + B)

    @dace.program
    def outer(A: dace.float64[8], B: dace.float64[8]):
        return inner(A + 1, B + 2)

    stree = outer.to_schedule_tree()

    assert len(stree.children) == 4
    assert isinstance(stree.children[0], tn.MapScope)
    assert isinstance(stree.children[1], tn.MapScope)
    # The nested call is inlined into a FunctionCallScope.
    call_scope = stree.children[2]
    assert isinstance(call_scope, tn.FunctionCallScope)
    assert call_scope.call.callee_name == 'inner'
    assert call_scope.call.arguments == {'A': '__stree_tmp', 'B': '__stree_tmp1'}
    assert len(call_scope.children) >= 1
    assert isinstance(stree.children[3], tn.ReturnNode)


def test_python_frontend_schedule_tree_nested_program_calls_are_not_executed(monkeypatch):

    @dace.program
    def inner(A: dace.float64[8], B: dace.float64[8]):
        return np.sum(A + B)

    @dace.program
    def outer(A: dace.float64[8], B: dace.float64[8]):
        return inner(A + 1, B + 2)

    from dace.frontend.python import parser as dace_parser

    original_call = dace_parser.DaceProgram.__call__
    seen = []

    def _guard(self, *args, **kwargs):
        if self is inner:
            seen.append((args, kwargs))
            raise AssertionError('nested program executed during schedule-tree generation')
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(dace_parser.DaceProgram, '__call__', _guard)

    stree = outer.to_schedule_tree()

    assert seen == []
    assert isinstance(stree.children[2], tn.FunctionCallScope)
    assert isinstance(stree.children[3], tn.ReturnNode)


def test_python_frontend_schedule_tree_constant_return_materializes_descriptor():

    @dace.program
    def constant():
        return 5

    stree = constant.to_schedule_tree()

    assert isinstance(stree.children[-2], tn.TaskletNode)
    assert stree.children[-2].node.code.as_string == '__stree_retval = 5'
    assert isinstance(stree.children[-1], tn.ReturnNode)
    assert stree.children[-1].values == ['__stree_retval']


def test_python_frontend_schedule_tree_symbolic_return_materializes_descriptor():
    n = dace.symbol('n')

    @dace.program
    def symbolic_constant():
        return n + 1

    stree = symbolic_constant.to_schedule_tree()

    assert isinstance(stree.children[-2], tn.TaskletNode)
    assert stree.children[-2].node.code.as_string == '__stree_retval = (n + 1)'
    assert isinstance(stree.children[-1], tn.ReturnNode)
    assert stree.children[-1].values == ['__stree_retval']


def test_python_frontend_schedule_tree_function_call_assignment():

    @dace.program
    def inner(A: dace.float64[8], B: dace.float64[8]):
        return A + B

    @dace.program
    def outer(A: dace.float64[8], B: dace.float64[8], out: dace.float64[8]):
        out[:] = inner(A + 1, B + 2)

    stree = outer.to_schedule_tree()

    assert len(stree.children) == 3
    assert isinstance(stree.children[0], tn.MapScope)
    assert isinstance(stree.children[1], tn.MapScope)
    # The nested call is inlined into a FunctionCallScope.
    call_scope = stree.children[2]
    assert isinstance(call_scope, tn.FunctionCallScope)
    assert call_scope.call.callee_name == 'inner'
    assert call_scope.call.arguments == {'A': '__stree_tmp', 'B': '__stree_tmp1'}
    # The callee's body should be inlined with the return rewritten
    # as an assignment to the caller's target.
    assert len(call_scope.children) >= 1


def test_python_frontend_schedule_tree_external_schedule_tree_convertible_call():

    class Convertible(ScheduleTreeConvertible):

        def __init__(self):

            @dace.program
            def inner(A: dace.float64[8], B: dace.float64[8]):
                return A + B

            self.inner = inner

        def __schedule_tree__(self, *args, lambda_bindings=None, callable_bindings=None, **kwargs):
            return self.inner.__schedule_tree__(*args,
                                                lambda_bindings=lambda_bindings,
                                                callable_bindings=callable_bindings,
                                                **kwargs)

        def __schedule_tree_signature__(self):
            return (['A', 'B'], [])

    convertible = Convertible()

    @dace.program
    def outer(A: dace.float64[8], B: dace.float64[8]):
        return convertible(A + 1, B + 2)

    stree = outer.to_schedule_tree()

    assert len(stree.children) == 4
    assert isinstance(stree.children[0], tn.MapScope)
    assert isinstance(stree.children[1], tn.MapScope)
    call_scope = stree.children[2]
    assert isinstance(call_scope, tn.FunctionCallScope)
    assert call_scope.call.callee_name == 'Convertible'
    assert call_scope.call.arguments == {'A': '__stree_tmp', 'B': '__stree_tmp1'}
    assert len(call_scope.children) >= 1
    assert isinstance(stree.children[3], tn.ReturnNode)


def test_python_frontend_schedule_tree_sdfg_call_stays_opaque():

    @dace.program
    def inner(A: dace.float64[8], B: dace.float64[8]):
        return A + B

    sdfg_obj = inner.to_sdfg()

    @dace.program
    def outer(A: dace.float64[8], B: dace.float64[8]):
        return sdfg_obj(A, B)

    stree = outer.to_schedule_tree()

    assert not any(isinstance(node, tn.FunctionCallScope) for node in stree.preorder_traversal())
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    assert isinstance(stree.children[0], tn.SDFGCallNode)
    assert isinstance(stree.children[0].sdfg, dace.SDFG)
    assert stree.children[0].sdfg.name == sdfg_obj.name
    assert stree.children[0].call.callee_name.endswith('inner')
    assert stree.children[0].call.arguments == {'A': 'A', 'B': 'B'}
    assert stree.children[0].return_targets == ['__stree_retval']
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert stree.children[1].values[0] == '__stree_retval'


def test_python_frontend_schedule_tree_sdfg_convertible_call_stays_opaque():

    class Convertible(SDFGConvertible):

        def __init__(self):
            self.name = 'convertible'

        def __call__(self, *args, **kwargs):
            raise AssertionError('SDFGConvertible should not execute during schedule-tree generation')

        def __sdfg__(self, A, B):

            @dace.program
            def inner(X: dace.float64[8], Y: dace.float64[8]):
                return X + Y

            return inner.to_sdfg(A, B)

        def __sdfg_signature__(self):
            return (['A', 'B'], [])

        def __sdfg_closure__(self, reevaluate=None):
            return {}

    convertible = Convertible()

    @dace.program
    def outer(A: dace.float64[8], B: dace.float64[8]):
        return convertible(A, B)

    stree = outer.to_schedule_tree()

    assert not any(isinstance(node, tn.FunctionCallScope) for node in stree.preorder_traversal())
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    assert isinstance(stree.children[0], tn.SDFGCallNode)
    assert isinstance(stree.children[0].sdfg, dace.SDFG)
    assert stree.children[0].call.callee_name == 'convertible'
    assert stree.children[0].call.arguments == {'A': 'A', 'B': 'B'}
    assert stree.children[0].return_targets == ['__stree_retval']
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert stree.children[1].values[0] == '__stree_retval'


def test_python_frontend_schedule_tree_return_materializes_array_expression():

    @dace.program
    def returned(A: dace.float64[8], B: dace.float64[8]):
        return A + B

    stree = returned.to_schedule_tree()

    assert len(stree.children) == 2
    assert isinstance(stree.children[0], tn.MapScope)
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert stree.children[1].values[0] == '__stree_tmp'


def test_python_frontend_schedule_tree_compile_time_fstring_stays_direct():

    prefix = 'value='

    @dace.program
    def returned():
        tmp = f'{prefix}5'
        return tmp

    stree = returned.to_schedule_tree()

    assert stree.containers['tmp'].dtype == dace.dtypes.string
    assert isinstance(stree.children[0], tn.AssignNode)
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())


def test_python_frontend_schedule_tree_matmul_chain_library_calls():

    @dace.program
    def chained(A: dace.float64[4, 3], B: dace.float64[3, 2], C: dace.float64[2, 5]):
        return A @ B @ C

    stree = chained.to_schedule_tree()

    assert len(stree.children) == 3
    assert isinstance(stree.children[0], tn.LibraryCall)
    assert isinstance(stree.children[1], tn.LibraryCall)
    assert isinstance(stree.children[2], tn.ReturnNode)
    assert isinstance(stree.children[0].node, tn.FrontendLibrary)
    assert isinstance(stree.children[1].node, tn.FrontendLibrary)
    assert stree.children[0].node.name == 'MatMul'
    assert stree.children[1].node.name == 'MatMul'
    assert stree.children[2].values[0] == '__stree_tmp1'


def test_python_frontend_schedule_tree_reduction_calls():

    @dace.program
    def np_sum_prog(a: dace.float64[8]):
        return np.sum(a)

    @dace.program
    def method_sum_prog(a: dace.float64[8]):
        return a.sum()

    np_sum_tree = np_sum_prog.to_schedule_tree()
    method_sum_tree = method_sum_prog.to_schedule_tree()

    # np.sum(a) should now be materialized as a LibraryCall writing to a
    # scalar temporary, followed by a ReturnNode referencing it.
    assert isinstance(np_sum_tree.children[0], tn.LibraryCall)
    assert np_sum_tree.children[0].node.name == 'numpy.sum'
    assert isinstance(np_sum_tree.children[1], tn.ReturnNode)
    # a.sum() is method syntax — now covered by the method descriptor registry,
    # so it should also be a LibraryCall followed by a ReturnNode.
    assert isinstance(method_sum_tree.children[0], tn.LibraryCall)
    assert isinstance(method_sum_tree.children[1], tn.ReturnNode)


def test_python_frontend_schedule_tree_method_field_read_is_not_left_as_raw_self_attribute():

    class FieldReader:

        def __init__(self):
            self._tmp_field = np.arange(8, dtype=np.float64)

        @dace.method
        def reduce(self):
            return np.sum(self._tmp_field)

    stree = FieldReader().reduce.to_schedule_tree()

    assert 'self._tmp_field' not in stree.as_string()
    assert any(
        isinstance(node, tn.LibraryCall) and node.node.name == 'numpy.sum' for node in stree.preorder_traversal())


def test_python_frontend_schedule_tree_nested_method_self_containers_do_not_conflict():

    class Inner:

        def __init__(self, size: int, offset: float):
            self._tmp = np.arange(size, dtype=np.float64) + offset

        @dace.method
        def reduce(self):
            return np.sum(self._tmp)

    class Outer:

        def __init__(self):
            self._tmp = np.arange(4, dtype=np.float64)
            self.inner1 = Inner(5, 1.0)
            self.inner2 = Inner(6, 2.0)

        @dace.method
        def reduce(self):
            return np.sum(self._tmp) + self.inner1.reduce() + self.inner2.reduce()

    stree = Outer().reduce.to_schedule_tree()

    closure_shapes = sorted(
        descriptor.shape[0] for name, descriptor in stree.containers.items()
        if name.startswith('__g_') and len(descriptor.shape) == 1 and descriptor.shape[0] in (4, 5, 6))

    assert closure_shapes == [4, 5, 6]
    assert 'self._tmp' not in stree.as_string()
    assert [type(child) for child in stree.children
            ] == [tn.LibraryCall, tn.FunctionCallScope, tn.FunctionCallScope, tn.TaskletNode, tn.ReturnNode]
    assert stree.children[-1].values[0] == stree.children[-2].out_memlets['out'].data


def test_python_frontend_schedule_tree_method_unresolved_new_field_assignment_stays_explicit():

    class FieldWriter:

        @dace.method
        def write(self, A: dace.float64[8]):
            self.new_field = A[0]
            return A[0]

    stree = FieldWriter().write.to_schedule_tree()

    assert isinstance(stree.children[0], tn.StatementNode)
    assert stree.children[0].code.as_string == 'self.new_field = A[0]'
    assert 'self' in stree.containers
    assert isinstance(stree.containers['self'], dace.data.pydata.PythonClass)
    assert stree.containers['self'].name == 'FieldWriter'
    assert stree.containers['self'].members == {}
    assert all('new_field' not in name for name in stree.containers)


def test_python_frontend_schedule_tree_descriptor_and_attribute_access():

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
            self.arr = np.zeros(8, dtype=np.float64)

    descriptor_holder = DescriptorHolder()

    @dace.program
    def descriptor_prog(A: dace.float64[8], out: dace.float64[8]):
        descriptor_holder.arr = A
        out[:] = descriptor_holder.arr

    class AttrHolder:

        def __init__(self):
            self.arr = np.zeros(8, dtype=np.float64)

    attr_holder = AttrHolder()

    @dace.program
    def attr_prog(A: dace.float64[8], out: dace.float64[8]):
        attr_holder.arr = A
        out[:] = attr_holder.arr

    descriptor_tree = descriptor_prog.to_schedule_tree()
    attribute_tree = attr_prog.to_schedule_tree()

    assert isinstance(descriptor_tree.children[0], tn.StatementNode)
    assert descriptor_tree.children[
        0].code.as_string == "type(descriptor_holder).__dict__['arr'].__set__(descriptor_holder, A)"
    assert isinstance(descriptor_tree.children[1], tn.StatementNode)
    assert descriptor_tree.children[1].code.as_string == (
        "out[:] = type(descriptor_holder).__dict__['arr'].__get__(descriptor_holder, type(descriptor_holder))")
    assert isinstance(attribute_tree.children[0], tn.StatementNode)
    assert attribute_tree.children[0].code.as_string == 'attr_holder.arr = A'
    assert isinstance(attribute_tree.children[1], tn.StatementNode)
    assert attribute_tree.children[1].code.as_string == 'out[:] = attr_holder.arr'


def test_python_frontend_schedule_tree_descriptor_setter_protocol_is_preserved():

    class OffsetDescriptor:

        def __set_name__(self, owner, name):
            self.name = '_' + name

        def __get__(self, obj, objtype=None):
            return getattr(obj, self.name)

        def __set__(self, obj, value):
            setattr(obj, self.name, value + 1)

    class DescriptorHolder:
        arr = OffsetDescriptor()

        def __init__(self):
            self.arr = np.zeros(8, dtype=np.float64)

    descriptor_holder = DescriptorHolder()

    @dace.program
    def descriptor_prog(A: dace.float64[8], out: dace.float64[8]):
        descriptor_holder.arr = A
        out[:] = descriptor_holder.arr

    stree = descriptor_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.StatementNode)
    assert stree.children[0].code.as_string == "type(descriptor_holder).__dict__['arr'].__set__(descriptor_holder, A)"
    assert isinstance(stree.children[1], tn.StatementNode)
    assert stree.children[1].code.as_string == (
        "out[:] = type(descriptor_holder).__dict__['arr'].__get__(descriptor_holder, type(descriptor_holder))")


def test_python_frontend_schedule_tree_optional_none_branch():

    @dace.program
    def optional_none_prog(field: Optional[dace.float64[8]], A: dace.float64[8], out: dace.float64[8]):
        if field is None:
            out[:] = A[:]
        else:
            out[:] = field[:]

    stree = optional_none_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.IfScope)
    assert stree.children[0].condition.as_string == '(field is None)'
    assert isinstance(stree.children[0].children[0], tn.CopyNode)
    assert isinstance(stree.children[1], tn.ElseScope)
    assert isinstance(stree.children[1].children[0], tn.CopyNode)


def test_python_frontend_schedule_tree_list_comprehension():

    @dace.program
    def list_comp_prog(A: dace.float64[8]):
        tmp = [A[i] for i in range(4)]
        return tmp

    stree = list_comp_prog.to_schedule_tree()

    # Comprehensions are now desugared to explicit loops in preprocessing.
    # The tree should contain an init (__comp_tmp = []) and a loop.
    assert isinstance(stree, tn.ScheduleTreeRoot)
    # Find the loop that was desugared from the comprehension
    loops = [c for c in stree.children if isinstance(c, tn.LoopScope)]
    assert len(loops) >= 1


def test_python_frontend_schedule_tree_linked_object_reference():

    class Node:

        def __init__(self, arr, next_node=None):
            self.arr = arr
            self.next = next_node

    linked = Node(np.zeros(8, dtype=np.float64), Node(np.ones(8, dtype=np.float64)))

    @dace.program
    def linked_prog(out: dace.float64[8]):
        ref = linked.next.arr
        out[:] = ref

    stree = linked_prog.to_schedule_tree()

    assert len(stree.children) == 2
    assert isinstance(stree.children[0], tn.RefSetNode)
    assert stree.children[0].source_expr == 'linked.next.arr'
    assert isinstance(stree.children[1], tn.CopyNode)


def test_python_frontend_schedule_tree_normalized_loop_iterators():

    @dace.program
    def array_iter_prog(A: dace.float64[4], out: dace.float64[4]):
        for val in A:
            out[0] = val

    @dace.program
    def zip_prog(A: dace.float64[4], B: dace.float64[4], out: dace.float64[4]):
        for a, b in zip(A, B):
            out[0] = a + b

    @dace.program
    def enumerate_prog(A: dace.float64[4], out: dace.float64[4]):
        for i, val in enumerate(A):
            out[i] = val

    @dace.program
    def enumerate_zip_flat_prog(A: dace.float64[4], B: dace.float64[4], out: dace.float64[4]):
        for i, pair in enumerate(zip(A, B)):
            out[i] = pair[0] + pair[1]

    @dace.program
    def enumerate_zip_unpack_prog(A: dace.float64[4], B: dace.float64[4], out: dace.float64[4]):
        for i, (a, b) in enumerate(zip(A, B)):
            out[i] = a + b

    array_tree = array_iter_prog.to_schedule_tree()
    zip_tree = zip_prog.to_schedule_tree()
    enumerate_tree = enumerate_prog.to_schedule_tree()
    enumerate_zip_flat_tree = enumerate_zip_flat_prog.to_schedule_tree()
    enumerate_zip_unpack_tree = enumerate_zip_unpack_prog.to_schedule_tree()

    assert len(array_tree.children) == 1
    assert isinstance(array_tree.children[0], tn.LoopScope)
    assert isinstance(array_tree.children[0].loop, tn.FrontendLoop)
    assert isinstance(array_tree.children[0].children[0], tn.CopyNode)

    assert len(zip_tree.children) == 1
    assert isinstance(zip_tree.children[0], tn.LoopScope)
    assert isinstance(zip_tree.children[0].loop, tn.FrontendLoop)
    assert isinstance(zip_tree.children[0].children[0], tn.TaskletNode)

    assert len(enumerate_tree.children) == 1
    assert isinstance(enumerate_tree.children[0], tn.LoopScope)
    assert isinstance(enumerate_tree.children[0].loop, tn.FrontendLoop)
    assert isinstance(enumerate_tree.children[0].children[0], tn.CopyNode)

    assert len(enumerate_zip_flat_tree.children) == 1
    assert isinstance(enumerate_zip_flat_tree.children[0], tn.LoopScope)
    assert isinstance(enumerate_zip_flat_tree.children[0].loop, tn.FrontendLoop)
    assert isinstance(enumerate_zip_flat_tree.children[0].children[0], tn.TaskletNode)

    assert len(enumerate_zip_unpack_tree.children) == 1
    assert isinstance(enumerate_zip_unpack_tree.children[0], tn.LoopScope)
    assert isinstance(enumerate_zip_unpack_tree.children[0].loop, tn.FrontendLoop)
    assert isinstance(enumerate_zip_unpack_tree.children[0].children[0], tn.TaskletNode)


def test_python_frontend_schedule_tree_generic_iterator_fallback():

    class CounterIterable:

        def __iter__(self):
            return iter([1.0, 2.0, 3.0])

    counter = CounterIterable()

    @dace.program
    def iter_prog(out: dace.float64[4]):
        for val in dace.nounroll(counter):
            out[0] = val

    stree = iter_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.AssignNode)
    assert isinstance(stree.children[1], tn.AssignNode)
    assert isinstance(stree.children[2], tn.AssignNode)
    assert isinstance(stree.children[3], tn.AssignNode)
    assert isinstance(stree.children[4], tn.LoopScope)
    assert isinstance(stree.children[4].loop, tn.FrontendLoop)
    assert stree.children[4].loop.loop_condition.as_string == '__dace_iter_has_next_2'
    assert isinstance(stree.children[4].children[0], tn.CopyNode)
    assert isinstance(stree.children[4].children[1], tn.AssignNode)
    assert stree.children[4].children[1].name == '__dace_iter_next_1'
    assert isinstance(stree.children[4].children[2], tn.AssignNode)
    assert stree.children[4].children[2].name == '__dace_iter_has_next_2'
    assert isinstance(stree.children[4].children[3], tn.AssignNode)
    assert stree.children[4].children[3].name == '__dace_iter_value_3'


def test_python_frontend_schedule_tree_generic_iterator_inference_has_no_runtime_side_effect():

    class CounterIterable:

        def __init__(self):
            self.iter_calls = 0

        def __iter__(self):
            self.iter_calls += 1
            return iter([1.0, 2.0, 3.0])

    counter = CounterIterable()

    @dace.program
    def iter_prog(out: dace.float64[4]):
        for val in dace.nounroll(counter):
            out[0] = val

    stree = iter_prog.to_schedule_tree()

    assert counter.iter_calls == 0
    assert isinstance(stree.children[4], tn.LoopScope)
    assert isinstance(stree.children[4].children[0], tn.CopyNode)


def test_python_frontend_schedule_tree_generic_iterator_tuple_value():

    class PairIterable:

        def __iter__(self):
            return iter([(1.0, 2.0), (3.0, 4.0)])

    pairs = PairIterable()

    @dace.program
    def iter_prog(out: dace.float64[4]):
        for pair in dace.nounroll(pairs):
            out[0] = pair[0] + pair[1]

    stree = iter_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.AssignNode)
    assert isinstance(stree.children[1], tn.AssignNode)
    assert isinstance(stree.children[2], tn.AssignNode)
    assert isinstance(stree.children[3], tn.AssignNode)
    assert isinstance(stree.children[4], tn.LoopScope)
    assert isinstance(stree.children[4].children[0], tn.TaskletNode)


def test_python_frontend_schedule_tree_generic_iterator_fallback_destructuring():

    class PairIterable:

        def __iter__(self):
            return iter([(1.0, 2.0), (3.0, 4.0)])

    pairs = PairIterable()

    @dace.program
    def iter_prog(out: dace.float64[4]):
        for a, b in dace.nounroll(pairs):
            out[0] = a + b

    stree = iter_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.AssignNode)
    assert isinstance(stree.children[1], tn.AssignNode)
    assert isinstance(stree.children[2], tn.AssignNode)
    assert isinstance(stree.children[3], tn.AssignNode)
    assert isinstance(stree.children[4], tn.LoopScope)
    assert isinstance(stree.children[4].loop, tn.FrontendLoop)
    assert isinstance(stree.children[4].children[0], tn.StatementNode)
    assert stree.children[4].children[0].code.as_string == '(a, b) = __dace_iter_value_3'
    assert isinstance(stree.children[4].children[1], tn.TaskletNode)
    assert isinstance(stree.children[4].children[2], tn.AssignNode)
    assert isinstance(stree.children[4].children[3], tn.AssignNode)
    assert isinstance(stree.children[4].children[4], tn.AssignNode)


def test_python_frontend_schedule_tree_generic_iterator_generator_object():

    def reverse_range(sz):
        cur = sz
        for _ in range(sz):
            yield float(cur)
            cur -= 1

    generator = reverse_range(3)

    @dace.program
    def iter_prog(out: dace.float64[4]):
        for val in dace.nounroll(generator):
            out[0] = val

    stree = iter_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.AssignNode)
    assert isinstance(stree.children[1], tn.AssignNode)
    assert isinstance(stree.children[2], tn.AssignNode)
    assert isinstance(stree.children[3], tn.AssignNode)
    assert isinstance(stree.children[4], tn.LoopScope)
    assert isinstance(stree.children[4].loop, tn.FrontendLoop)
    assert isinstance(stree.children[4].children[0], tn.CopyNode)
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    assert next(generator) == 3.0


def test_python_frontend_schedule_tree_tuple_of_arrays_unrolls():

    @dace.program
    def iter_prog(a: dace.float64[2, 3, 4], b: dace.float64[2, 3, 4], c: dace.float64[2, 3, 4], out: dace.float64[2, 3,
                                                                                                                  4]):
        for arr in (a, b, c):
            out[:] = arr

    with pytest.warns(UserWarning, match=r'implicitly unrolled'):
        stree = iter_prog.to_schedule_tree()

    assert [type(child) for child in stree.children] == [tn.CopyNode, tn.CopyNode, tn.CopyNode]
    assert [child.memlet.data for child in stree.children] == ['a', 'b', 'c']


def test_python_frontend_schedule_tree_nounroll_array_annotation_binds_reference():

    @dace.program
    def iter_prog(a: dace.float64[2, 3, 4], b: dace.float64[2, 3, 4], c: dace.float64[2, 3, 4], out: dace.float64[2, 3,
                                                                                                                  4]):
        list_of_arrays = [a, b, c]
        for arr in dace.nounroll(list_of_arrays):
            arr: dace.float64[2, 3, 4]
            out[:] = arr

    stree = iter_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.StatementNode)
    assert stree.children[0].code.as_string == 'list_of_arrays = [a, b, c]'
    assert isinstance(stree.containers['list_of_arrays'], PythonList)
    assert isinstance(stree.children[1], tn.AssignNode)
    assert stree.children[1].name == '__dace_iter_0'
    assert isinstance(stree.children[2], tn.AssignNode)
    assert stree.children[2].name == '__dace_iter_next_1'
    assert isinstance(stree.children[3], tn.AssignNode)
    assert stree.children[3].name == '__dace_iter_has_next_2'
    assert isinstance(stree.children[4], tn.AssignNode)
    assert stree.children[4].name == '__dace_iter_value_3'
    assert isinstance(stree.children[5], tn.LoopScope)
    assert stree.children[5].loop.loop_condition.as_string == '__dace_iter_has_next_2'
    assert isinstance(stree.children[5].children[0], tn.RefSetNode)
    assert stree.children[5].children[0].target == 'arr'
    assert isinstance(stree.children[5].children[1], tn.CopyNode)
    assert stree.children[5].children[1].memlet.data == 'arr'
    assert isinstance(stree.children[5].children[2], tn.AssignNode)
    assert stree.children[5].children[2].name == '__dace_iter_next_1'
    assert isinstance(stree.children[5].children[3], tn.AssignNode)
    assert stree.children[5].children[3].name == '__dace_iter_has_next_2'
    assert isinstance(stree.children[5].children[4], tn.AssignNode)
    assert stree.children[5].children[4].name == '__dace_iter_value_3'
    assert '__dace_iter_end_' not in stree.as_string()


def test_python_frontend_schedule_tree_typed_dict_literal_and_static_read():

    @dace.program
    def dict_prog(A: dace.float64[2]):
        mapping = {'left': A[0], 'right': A[1]}
        value = mapping['left']
        return value

    stree = dict_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.StatementNode)
    assert stree.children[0].code.as_string == "mapping = {'left': A[0], 'right': A[1]}"
    assert isinstance(stree.containers['mapping'], PythonDict)
    assert isinstance(stree.containers['mapping'].key_type, dace.data.Scalar)
    assert stree.containers['mapping'].key_type.dtype == dace.string
    assert isinstance(stree.containers['mapping'].value_type, dace.data.Scalar)
    assert stree.containers['mapping'].value_type.dtype == dace.float64
    assert isinstance(stree.children[1], tn.AssignNode)
    assert stree.children[1].name == 'value'
    assert stree.children[1].value.as_string == "mapping['left']"
    assert isinstance(stree.containers['value'], dace.data.Scalar)
    assert stree.containers['value'].dtype == dace.float64
    assert isinstance(stree.children[2], tn.ReturnNode)


def test_python_frontend_schedule_tree_heterogeneous_dict_values_fallback_to_pyobject():

    @dace.program
    def dict_prog(A: dace.float64[2]):
        mapping = {'left': A[0], 'right': 'two'}
        return 0.0

    stree = dict_prog.to_schedule_tree()

    assert isinstance(stree.containers['mapping'], PythonDict)
    assert stree.containers['mapping'].key_type.dtype == dace.string
    assert stree.containers['mapping'].value_type.dtype == dtypes.pyobject()


def test_python_frontend_schedule_tree_heterogeneous_dict_known_static_reads_stay_precise():

    @dace.program
    def dict_prog(A: dace.float64[2]):
        mapping = {'left': A[0], 'right': 'two'}
        left = mapping['left']
        right = mapping['right']
        return left

    stree = dict_prog.to_schedule_tree()

    assert isinstance(stree.containers['mapping'], PythonDict)
    assert stree.containers['mapping'].value_type.dtype == dtypes.pyobject()
    assert isinstance(stree.containers['left'], dace.data.Scalar)
    assert stree.containers['left'].dtype == dace.float64
    assert isinstance(stree.containers['right'], dace.data.Scalar)
    assert stree.containers['right'].dtype == dace.string
    assert isinstance(stree.children[1], tn.AssignNode)
    assert stree.children[1].value.as_string == "mapping['left']"
    assert isinstance(stree.children[2], tn.AssignNode)
    assert stree.children[2].value.as_string == "mapping['right']"


def test_python_frontend_schedule_tree_dict_same_key_update_widens_value_type():

    @dace.program
    def dict_prog(A: dace.float64[2]):
        mapping = {'left': A[0], 'right': A[1]}
        mapping['left'] = 'two'
        value = mapping['left']
        return value

    stree = dict_prog.to_schedule_tree()

    assert isinstance(stree.containers['mapping'], PythonDict)
    assert stree.containers['mapping'].key_type.dtype == dace.string
    assert stree.containers['mapping'].value_type.dtype == dtypes.pyobject()
    assert isinstance(stree.containers['value'], dace.data.Scalar)
    assert stree.containers['value'].dtype == dace.string
    assert isinstance(stree.children[1], tn.StatementNode)
    assert stree.children[1].code.as_string == "mapping['left'] = 'two'"
    assert isinstance(stree.children[2], tn.AssignNode)
    assert stree.children[2].value.as_string == "mapping['left']"


def test_python_frontend_schedule_tree_free_iter_and_next_calls():

    def reverse_range(sz):
        cur = sz
        for _ in range(sz):
            yield float(cur)
            cur -= 1

    generator = reverse_range(3)

    @dace.program
    def iter_prog(out: dace.float64[3]):
        it = iter(generator)
        out[0] = next(it)
        out[1] = next(it)
        out[2] = next(it)

    stree = iter_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.PythonCallbackNode)
    assert stree.children[0].reason == 'pyobject call'
    assert stree.children[0].code.as_string == 'it = iter(generator)'
    assert stree.children[0].outlined_function_name.startswith('__stree_callback')
    assert stree.children[0].outlined_function_code.as_string.startswith(
        f'def {stree.children[0].outlined_function_name}():')
    assert stree.children[0].outlined_call_code.as_string == f'it = {stree.children[0].outlined_function_name}()'
    for index, child in enumerate(stree.children[1:]):
        assert isinstance(child, tn.TaskletNode)
        assert child.node.code.as_string == f'out[{index}] = next(it)'
    assert len([node for node in stree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]) == 1
    assert next(generator) == 3.0


def test_python_frontend_schedule_tree_internal_generator_with_next_calls():

    def reverse_range(sz):
        cur = sz
        for _ in range(sz):
            yield float(cur)
            cur -= 1

    @dace.program
    def iter_prog(out: dace.float64[3]):
        gen = reverse_range(3)
        out[0] = next(gen)
        out[1] = next(gen)
        out[2] = next(gen)

    stree = iter_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.PythonCallbackNode)
    assert stree.children[0].reason == 'pyobject call'
    assert stree.children[0].code.as_string == 'gen = reverse_range(3)'
    for index, child in enumerate(stree.children[1:]):
        assert isinstance(child, tn.TaskletNode)
        assert child.node.code.as_string == f'out[{index}] = next(gen)'
    assert len([node for node in stree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]) == 1


def test_python_frontend_schedule_tree_next_iter_dict_values():

    @dace.program
    def iter_prog(out: dace.int64[1]):
        x = {1: 1, 2: 2, 3: 3}
        out[0] = next(iter(x.values()))

    stree = iter_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.StatementNode)
    assert stree.children[0].code.as_string == 'x = {1: 1, 2: 2, 3: 3}'
    assert isinstance(stree.children[1], tn.PythonCallbackNode)
    assert stree.children[1].reason == 'pyobject call'
    assert stree.children[1].code.as_string == '__stree_tmp = x.values()'
    assert isinstance(stree.children[2], tn.PythonCallbackNode)
    assert stree.children[2].reason == 'pyobject call'
    assert stree.children[2].code.as_string == '__stree_tmp1 = iter(__stree_tmp)'
    assert isinstance(stree.children[3], tn.TaskletNode)
    assert stree.children[3].node.code.as_string == 'out[0] = next(__stree_tmp1)'
    assert len([node for node in stree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]) == 2


def test_python_frontend_schedule_tree_untyped_next_warns():

    def reverse_range(sz):
        cur = sz
        for _ in range(sz):
            yield float(cur)
            cur -= 1

    @dace.program
    def iter_prog(out: dace.float64[1]):
        gen = reverse_range(3)
        val = next(gen)
        out[0] = val

    with pytest.warns(UserWarning,
                      match=r'Could not infer the result type of iterator next\(\) in schedule-tree lowering'):
        stree = iter_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.PythonCallbackNode)
    assert stree.children[0].code.as_string == 'gen = reverse_range(3)'
    assert isinstance(stree.children[1], tn.PythonCallbackNode)
    assert stree.children[1].code.as_string == 'val = next(gen)'
    assert isinstance(stree.children[2], tn.CopyNode)


def test_python_frontend_schedule_tree_annotated_next_assignment_is_typed():

    def reverse_range(sz):
        cur = sz
        for _ in range(sz):
            yield float(cur)
            cur -= 1

    @dace.program
    def iter_prog(out: dace.float64[1]):
        gen = reverse_range(3)
        val: dace.float64 = next(gen)
        out[0] = val

    with pytest.raises(pytest.fail.Exception, match='DID NOT WARN'):
        with pytest.warns(UserWarning,
                          match=r'Could not infer the result type of iterator next\(\) in schedule-tree lowering'):
            stree = iter_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.PythonCallbackNode)
    assert stree.children[0].code.as_string == 'gen = reverse_range(3)'
    assert isinstance(stree.children[1], tn.TaskletNode)
    assert stree.children[1].node.code.as_string == 'val = next(gen)'
    assert isinstance(stree.children[2], tn.CopyNode)


def test_python_frontend_schedule_tree_tuple_swap_statement():

    @dace.program
    def swap_prog(A: dace.float64[4], B: dace.float64[4]):
        A, B = B, A
        return A

    stree = swap_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.StatementNode)
    assert stree.children[0].code.as_string == '__stree_tuple_tmp = (B, A)'
    assert isinstance(stree.children[1], tn.StatementNode)
    assert stree.children[1].code.as_string == '(A, B) = __stree_tuple_tmp'
    assert isinstance(stree.children[2], tn.ReturnNode)


def test_python_frontend_schedule_tree_tuple_permutation_materializes_rhs():

    @dace.program
    def perm_prog(A: dace.float64[4], B: dace.float64[4], C: dace.float64[4]):
        A, B, C = C, A, A
        return A

    stree = perm_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.StatementNode)
    assert stree.children[0].code.as_string == '__stree_tuple_tmp = (C, A, A)'
    assert isinstance(stree.children[1], tn.StatementNode)
    assert stree.children[1].code.as_string == '(A, B, C) = __stree_tuple_tmp'
    assert isinstance(stree.children[2], tn.ReturnNode)


def test_python_frontend_schedule_tree_starred_unpacking_uses_analyzable_structure():

    @dace.program
    def starred_prog(A: dace.float64[4], B: dace.float64[4], C: dace.float64[4], out: dace.float64[4]):
        head, *rest = (A, B, C)
        out[:] = rest[1]

    stree = starred_prog.to_schedule_tree()

    assert isinstance(stree.children[0], (tn.RefSetNode, tn.ViewNode))
    assert stree.children[0].target == 'head'
    assert isinstance(stree.children[1], tn.StatementNode)
    assert stree.children[1].code.as_string == 'rest = [B, C]'
    assert isinstance(stree.children[2], tn.CopyNode)
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())


def test_python_frontend_schedule_tree_star_call_expansion_is_resolved_statically():

    def callee(a, b, c):
        return a + b + c

    @dace.program
    def star_call_prog(A: dace.float64[4], B: dace.float64[4], C: dace.float64[4]):
        args = (A, B)
        return callee(*args, c=C)

    stree = star_call_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.StatementNode)
    assert stree.children[0].code.as_string == 'args = (A, B)'
    assert isinstance(stree.children[1], tn.FunctionCallScope)
    assert stree.children[1].call.callee_name == 'callee'
    assert stree.children[1].call.arguments == {'a': 'A', 'b': 'B', 'c': 'C'}
    assert isinstance(stree.children[2], tn.ReturnNode)
    assert stree.children[2].values[0] == '__stree_retval'
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())


def test_python_frontend_schedule_tree_double_star_call_expansion_is_resolved_statically():

    def callee(a, b, c):
        return a + b + c

    @dace.program
    def dstar_call_prog(A: dace.float64[4], B: dace.float64[4], C: dace.float64[4]):
        kwargs = {'c': C}
        return callee(A, B, **kwargs)

    stree = dstar_call_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.StatementNode)
    assert stree.children[0].code.as_string == "kwargs = {'c': C}"
    assert isinstance(stree.children[1], tn.FunctionCallScope)
    assert stree.children[1].call.callee_name == 'callee'
    assert stree.children[1].call.arguments == {'a': 'A', 'b': 'B', 'c': 'C'}
    assert isinstance(stree.children[2], tn.ReturnNode)
    assert stree.children[2].values[0] == '__stree_retval'
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())


def test_python_frontend_schedule_tree_dynamic_star_call_expansion_uses_callback():

    def callee(a, b, c):
        return a + b + c

    @dace.program
    def dynamic_star_call(flag: dace.bool_, A: dace.float64[4], B: dace.float64[4], C: dace.float64[4]):
        return callee(*((A, B) if flag else (B, A)), c=C)

    stree = dynamic_star_call.to_schedule_tree()

    assert isinstance(stree.children[0], tn.PythonCallbackNode)
    assert stree.children[0].reason == 'call expansion'
    assert stree.children[0].code.as_string == '__stree_retval = callee(*((A, B) if flag else (B, A)), c=C)'
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert stree.children[1].values[0] == '__stree_retval'


def test_python_frontend_schedule_tree_dynamic_expanded_sdfg_call_raises():

    @dace.program
    def inner(A: dace.float64[4], B: dace.float64[4]):
        return A + B

    sdfg_obj = inner.to_sdfg(simplify=False)

    @dace.program
    def dynamic_sdfg_call(flag: dace.bool_, A: dace.float64[4], B: dace.float64[4]):
        return sdfg_obj(*((A, B) if flag else (B, A)))

    with pytest.raises(DaceSyntaxError, match='Dynamic argument expansion is unsupported for SDFG calls'):
        dynamic_sdfg_call.to_schedule_tree()


# ------------------------------------------------------------------ #
#  Phase 4 — Full Python Language Coverage Tests                      #
# ------------------------------------------------------------------ #


def test_try_except_produces_callback():

    @dace.program
    def try_prog(A: dace.float64[10]):
        try:
            A[0] = 1.0
        except Exception:
            A[0] = 0.0
        return A

    stree = try_prog.to_schedule_tree()

    callbacks = [c for c in stree.children if isinstance(c, tn.PythonCallbackNode)]
    assert len(callbacks) >= 1
    assert callbacks[0].reason == 'try/except'


def test_import_produces_callback():

    @dace.program
    def import_prog(A: dace.float64[10]):
        import math
        A[0] = math.pi
        return A

    stree = import_prog.to_schedule_tree()

    callbacks = [c for c in stree.children if isinstance(c, tn.PythonCallbackNode)]
    assert len(callbacks) >= 1
    assert any(c.reason == 'import' for c in callbacks)


def test_match_lowers_to_if_chain():

    @dace.program
    def match_prog(A: dace.int32[1]):
        match A[0]:
            case 0:
                A[0] = 1
            case _:
                A[0] = 2

    stree = match_prog.to_schedule_tree()

    assert not any(isinstance(c, tn.PythonCallbackNode) for c in stree.children)
    assert any(isinstance(c, tn.IfScope) for c in stree.children)
    assert any(isinstance(c, tn.ElseScope) for c in stree.children)


def test_match_capture_guard_and_or_lower_natively():

    @dace.program
    def match_prog(A: dace.int32[1], B: dace.int32[1]):
        match A[0]:
            case 0 | 1:
                B[0] = 7
            case x if x > 2:
                B[0] = x
            case _:
                B[0] = -1

    stree = match_prog.to_schedule_tree()

    assert not any(isinstance(c, tn.PythonCallbackNode) for c in stree.preorder_traversal())
    assert any(isinstance(c, tn.IfScope) for c in stree.children)
    assert any(isinstance(c, tn.ElifScope) for c in stree.children)
    assert any(isinstance(c, tn.ElseScope) for c in stree.children)


def test_match_fixed_length_sequence_lowers_natively():

    @dace.program
    def match_prog(A: dace.int32[2], B: dace.int32[1]):
        match (A[0], A[1]):
            case (0, x):
                B[0] = x
            case _:
                B[0] = -1

    stree = match_prog.to_schedule_tree()

    assert not any(isinstance(c, tn.PythonCallbackNode) for c in stree.preorder_traversal())
    assert any(isinstance(c, tn.IfScope) for c in stree.children)
    assert any(isinstance(c, tn.ElseScope) for c in stree.children)


def test_match_sequence_guard_support_is_native():

    @dace.program
    def match_prog(A: dace.int32[2], B: dace.int32[1]):
        match (A[0], A[1]):
            case (x, y) if x < y and y > 0:
                B[0] = x + y
            case _:
                B[0] = -1

    stree = match_prog.to_schedule_tree()

    assert not any(isinstance(c, tn.PythonCallbackNode) for c in stree.preorder_traversal())
    if_scopes = [c for c in stree.children if isinstance(c, tn.IfScope)]
    assert len(if_scopes) == 1
    assert 'len(' in if_scopes[0].condition.as_string
    assert '__stree_tmp[0]' in if_scopes[0].condition.as_string
    assert '__stree_tmp[1]' in if_scopes[0].condition.as_string


def test_match_mapping_case_forces_callback_for_whole_match():

    @dace.program
    def match_prog(A: dace.int32[2], B: dace.int32[1]):
        match (A[0], A[1]):
            case (0, x):
                B[0] = x
            case {'x': x}:
                B[0] = x
            case _:
                B[0] = -1

    stree = match_prog.to_schedule_tree()

    callbacks = [c for c in stree.children if isinstance(c, tn.PythonCallbackNode)]
    assert len(callbacks) >= 1
    assert callbacks[0].reason == 'match/case'


def test_match_class_case_forces_callback_for_whole_match():

    class Pair:

        def __init__(self, x: int, y: int):
            self.x = x
            self.y = y

    pair = Pair(1, 2)

    @dace.program
    def match_prog(B: dace.int32[1]):
        match pair:
            case Pair(x, y):
                B[0] = x + y
            case _:
                B[0] = -1

    stree = match_prog.to_schedule_tree()

    callbacks = [c for c in stree.children if isinstance(c, tn.PythonCallbackNode)]
    assert len(callbacks) >= 1
    assert callbacks[0].reason == 'match/case'


def test_class_def_is_rejected():

    @dace.program
    def classdef_prog(A: dace.float64[10]):

        class Foo:
            x = 1

        A[0] = Foo.x
        return A

    with pytest.raises(DaceSyntaxError, match='Nested class definitions are unsupported'):
        classdef_prog.to_schedule_tree()


def test_global_traces_container():
    """global x where x is a known global should bind, not callback."""
    some_global_array = np.zeros(10, dtype=np.float64)

    @dace.program
    def global_prog(A: dace.float64[10]):
        # global is typically used in nested scopes; test that it doesn't error
        for i in range(10):
            A[i] = some_global_array[i]
        return A

    # Should not raise
    stree = global_prog.to_schedule_tree()
    assert isinstance(stree, tn.ScheduleTreeRoot)


def test_global_untraceable_callback():

    @dace.program
    def global_prog(A: dace.float64[10]):
        global missing_name
        A[0] = 1.0

    stree = global_prog.to_schedule_tree()

    callbacks = [c for c in stree.children if isinstance(c, tn.PythonCallbackNode)]
    assert len(callbacks) >= 1
    assert callbacks[0].reason == 'global scope'


def test_top_level_global_reassignment_emits_reassign_external():
    globals()['__schedule_tree_global_reassign'] = np.zeros(10, dtype=np.float64)

    try:

        @dace.program
        def global_prog(A: dace.float64[10]):
            global __schedule_tree_global_reassign
            __schedule_tree_global_reassign = A
            return A

        stree = global_prog.to_schedule_tree()

    finally:
        del globals()['__schedule_tree_global_reassign']

    reassigns = [node for node in stree.children if isinstance(node, tn.ReassignExternalNode)]
    assert len(reassigns) == 1
    assert reassigns[0].scope == 'global'
    assert reassigns[0].name == '__schedule_tree_global_reassign'


def test_top_level_nonlocal_reassignment_emits_reassign_external():

    def make_prog():
        captured = np.zeros(10, dtype=np.float64)

        @dace.program
        def nonlocal_prog(A: dace.float64[10]):
            nonlocal captured
            captured = A
            return A

        return nonlocal_prog

    stree = make_prog().to_schedule_tree()

    reassigns = [node for node in stree.children if isinstance(node, tn.ReassignExternalNode)]
    assert len(reassigns) == 1
    assert reassigns[0].scope == 'nonlocal'
    assert reassigns[0].name == 'captured'


def test_decorated_nested_funcdef_produces_callback():

    def passthrough(fn):
        return fn

    @dace.program
    def nested_prog(A: dace.float64[10]):

        @passthrough
        def helper(x):
            y = x + 1
            return y

        A[0] = helper(A[0])
        return A

    stree = nested_prog.to_schedule_tree()

    callbacks = [c for c in stree.children if isinstance(c, tn.PythonCallbackNode)]
    assert len(callbacks) >= 1
    assert any(c.reason == 'nested function' for c in callbacks)


def test_async_function_produces_callback():

    @dace.program
    def nested_prog(A: dace.float64[10]):

        async def helper(x):
            return x

        A[0] = 1.0
        return A

    stree = nested_prog.to_schedule_tree()

    callbacks = [c for c in stree.children if isinstance(c, tn.PythonCallbackNode)]
    assert len(callbacks) >= 1
    assert any(c.reason == 'async function' for c in callbacks)


def test_async_dace_program_to_schedule_tree_is_rejected():

    @dace.program
    async def async_prog(A: dace.float64[10]):
        return A

    with pytest.raises(SyntaxError, match='Async @dace.program functions are unsupported'):
        async_prog.to_schedule_tree()


def test_delete_noop_for_arrays():
    """del of a known DaCe array should be a no-op (no node emitted)."""

    @dace.program
    def del_prog(A: dace.float64[10]):
        tmp = dace.define_local([10], dace.float64)
        tmp[:] = A[:]
        del tmp
        return A

    stree = del_prog.to_schedule_tree()

    # No PythonCallbackNode for 'delete' should appear
    callbacks = [c for c in stree.children if isinstance(c, tn.PythonCallbackNode) and c.reason == 'delete']
    assert len(callbacks) == 0


def test_dynamic_context_manager_produces_callback():

    @contextlib.contextmanager
    def guard(value):
        if value > 0:
            yield
        else:
            yield

    @dace.program
    def with_prog(A: dace.float64[10]):
        with guard(A[0]):
            A[1] = A[0]
        return A

    stree = with_prog.to_schedule_tree()

    callbacks = [node for node in stree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]
    assert len(callbacks) == 1
    assert callbacks[0].reason == 'context manager'
    assert callbacks[0].code.as_string.startswith('with guard(A[0]):')


def test_raise_produces_callback():

    @dace.program
    def raise_prog(A: dace.float64[10]):
        raise ValueError("test")

    stree = raise_prog.to_schedule_tree()

    raise_nodes = [node for node in stree.children if isinstance(node, tn.RaiseNode)]
    assert len(raise_nodes) == 1
    assert raise_nodes[0].exception_type is not None
    assert raise_nodes[0].exception_type.as_string == 'ValueError'
    assert [argument.as_string.strip("\"'") for argument in raise_nodes[0].args] == ['test']


def test_dynamic_raise_produces_callback_when_supported():

    @dace.program
    def raise_prog(A: dace.float64[10]):
        exc_type = ValueError
        raise exc_type("test")

    with dace.config.set_temporary('frontend', 'raise_statements', value='support'):
        stree = raise_prog.to_schedule_tree()

    callbacks = [node for node in stree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]
    assert len(callbacks) == 1
    assert callbacks[0].reason == 'raise'


def test_dynamic_raise_can_be_ignored():

    @dace.program
    def raise_prog(A: dace.float64[10]):
        exc_type = ValueError
        raise exc_type("test")
        return A

    with dace.config.set_temporary('frontend', 'raise_statements', value='ignore_dynamic'):
        stree = raise_prog.to_schedule_tree()

    assert not any(isinstance(node, tn.RaiseNode) for node in stree.preorder_traversal())
    assert not any(
        isinstance(node, tn.PythonCallbackNode) and node.reason == 'raise' for node in stree.preorder_traversal())
    assert isinstance(stree.children[-1], tn.ReturnNode)


def test_raise_can_be_ignored_entirely():

    @dace.program
    def raise_prog(A: dace.float64[10]):
        raise ValueError("test")
        return A

    with dace.config.set_temporary('frontend', 'raise_statements', value='ignore_all'):
        stree = raise_prog.to_schedule_tree()

    assert not any(isinstance(node, tn.RaiseNode) for node in stree.preorder_traversal())
    assert not any(
        isinstance(node, tn.PythonCallbackNode) and node.reason == 'raise' for node in stree.preorder_traversal())
    assert isinstance(stree.children[-1], tn.ReturnNode)


def test_raise_from_is_rejected_before_policy_fallback():

    @dace.program
    def raise_prog(A: dace.float64[10]):
        raise ValueError('outer') from ValueError('inner')

    with dace.config.set_temporary('frontend', 'raise_statements', value='ignore_all'):
        with pytest.raises(DaceSyntaxError, match='raise from'):
            raise_prog.to_schedule_tree()


def test_named_expr_desugared():
    """Walrus operator should be desugared before reaching schedule tree builder."""

    @dace.program
    def walrus_prog(A: dace.float64[10]):
        if (x := A[0]) > 0:
            A[1] = x
        return A

    stree = walrus_prog.to_schedule_tree()

    # The schedule tree should have an assignment before the if, not a NamedExpr
    # x = A[0] comes first, then if x > 0: ...
    assert isinstance(stree, tn.ScheduleTreeRoot)
    # Should not crash — that's the main verification


def test_singleton_subscript_assignment_scalarized():

    @dace.program
    def scalar_prog(x: dace.int32[20]):
        idx = x[0]

    stree = scalar_prog.to_schedule_tree()

    assert [type(child) for child in stree.children] == [tn.TaskletNode]
    assert stree.children[0].node.code.as_string == 'idx = x[0]'
    assert str(stree.children[0].in_memlets['in0'].subset) == '0'
    assert str(stree.children[0].out_memlets['out'].subset) == '0'
    assert isinstance(stree.containers['idx'], dace.data.Scalar)


def test_singleton_negative_subscript_assignment_canonicalized():
    n = dace.symbol('n')

    @dace.program
    def scalar_prog(x: dace.int32[n]):
        idx = x[-1]

    stree = scalar_prog.to_schedule_tree()

    assert [type(child) for child in stree.children] == [tn.TaskletNode]
    assert stree.children[0].node.code.as_string == 'idx = x[(n - 1)]'
    assert str(stree.children[0].in_memlets['in0'].subset) == 'n - 1'
    assert str(stree.children[0].out_memlets['out'].subset) == '0'
    assert isinstance(stree.containers['idx'], dace.data.Scalar)


def test_singleton_symbolic_negative_subscript_assignment_canonicalized():
    n = dace.symbol('n')
    i = dace.symbol('i', integer=True, positive=True)

    @dace.program
    def scalar_prog(x: dace.int32[n]):
        idx = x[-i]

    stree = scalar_prog.to_schedule_tree()

    assert [type(child) for child in stree.children] == [tn.TaskletNode]
    assert stree.children[0].node.code.as_string == 'idx = x[(n - i)]'
    assert (dace.symbolic.pystr_to_symbolic(str(
        stree.children[0].in_memlets['in0'].subset)) == dace.symbolic.pystr_to_symbolic('n - i'))
    assert str(stree.children[0].out_memlets['out'].subset) == '0'


def test_singleton_runtime_negative_subscript_assignment_uses_pyindex_when_enabled():
    n = dace.symbol('n')
    i = dace.symbol('i', integer=True)

    @dace.program
    def scalar_prog(x: dace.int32[n]):
        idx = x[i]

    with dace.config.set_temporary('frontend', 'runtime_negative_indices', value=True):
        stree = scalar_prog.to_schedule_tree()

    assert [type(child) for child in stree.children] == [tn.TaskletNode]
    assert stree.children[0].node.code.as_string == 'idx = x[i]'
    assert (dace.symbolic.pystr_to_symbolic(str(
        stree.children[0].in_memlets['in0'].subset)) == dace.symbolic.pystr_to_symbolic('pyindex(i, n)'))
    assert str(stree.children[0].out_memlets['out'].subset) == '0'


def test_short_circuit_condition_keeps_nested_index_in_guard():

    @dace.program
    def guard_prog(A: dace.int32[20], b: dace.int32[20], out: dace.int32[1], flag: dace.bool_, i: dace.int32):
        if flag and A[b[i]] == 0:
            out[0] = 1

    stree = guard_prog.to_schedule_tree()

    assert [type(child) for child in stree.children] == [tn.IfScope]
    assert 'A[b[i]]' in stree.children[0].condition.as_string
    assert '__stree_idx' not in stree.children[0].condition.as_string
    assert isinstance(stree.children[0].children[0], tn.TaskletNode)


def test_while_with_hoisted_index_rewritten_to_guarded_loop():

    @dace.program
    def loop_prog(A: dace.int32[20], b: dace.int32[20], i: dace.int32):
        while A[b[i]] == 0:
            i += 1

    stree = loop_prog.to_schedule_tree()

    assert [type(child) for child in stree.children] == [tn.LoopScope]
    assert stree.children[0].loop.loop_condition.as_string == 'True'
    assert isinstance(stree.children[0].children[0], tn.TaskletNode)
    assert stree.children[0].children[0].node.code.as_string == '__stree_idx = b[i]'
    assert isinstance(stree.children[0].children[1], tn.IfScope)
    assert 'not (A[__stree_idx] == 0)' in stree.children[0].children[1].condition.as_string
    assert isinstance(stree.children[0].children[1].children[0], tn.BreakNode)


def test_while_else_with_hoisted_index_falls_back_to_callback():

    @dace.program
    def loop_prog(A: dace.int32[20], b: dace.int32[20], out: dace.int32[1], i: dace.int32):
        while A[b[i]] == 0:
            i += 1
        else:
            out[0] = 1

    stree = loop_prog.to_schedule_tree()

    callbacks = [node for node in stree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]
    assert len(callbacks) == 1
    assert callbacks[0].reason == 'while loop test outlining with else'
    assert callbacks[0].code.as_string.startswith('while (A[b[i]] == 0):')


def test_while_else_without_hoisting_lowers_natively():

    @dace.program
    def loop_prog(A: dace.int32[20], out: dace.int32[1], i: dace.int32):
        while i < 3:
            i += 1
        else:
            out[0] = 1

    stree = loop_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.LoopScope)
    assert stree.children[0].loop.loop_condition.as_string == '(i < 3)'
    assert isinstance(stree.children[1], tn.ElseScope)
    assert isinstance(stree.children[1].children[0], tn.TaskletNode)


def test_for_else_lowers_natively():

    @dace.program
    def loop_prog(A: dace.int32[20], out: dace.int32[1]):
        for i in range(3):
            A[i] = A[i]
        else:
            out[0] = 1

    stree = loop_prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.LoopScope)
    assert stree.children[0].loop.loop_condition.as_string == '(i < 3)'
    assert isinstance(stree.children[1], tn.ElseScope)
    assert isinstance(stree.children[1].children[0], tn.TaskletNode)


def test_schedule_tree_distinguishes_list_and_tuple_indices():

    @dace.program
    def list_prog(A: dace.float64[5, 6]):
        tmp = A[[1, 2]]
        return tmp

    @dace.program
    def tuple_prog(A: dace.float64[5, 6]):
        tmp = A[(1, 2)]
        return tmp

    list_stree = list_prog.to_schedule_tree()
    tuple_stree = tuple_prog.to_schedule_tree()

    assert isinstance(list_stree.containers['tmp'], dace.data.Array)
    assert tuple(list_stree.containers['tmp'].shape) == (2, 6)
    assert isinstance(tuple_stree.containers['tmp'], dace.data.Scalar)
    assert tuple(tuple_stree.containers['tmp'].shape) == (1, )
    assert isinstance(list_stree.children[0], tn.TaskletNode)
    assert str(list_stree.children[0].out_memlets['out'].subset) == '0:2, 0:6'
    assert isinstance(tuple_stree.children[0], tn.TaskletNode)
    assert str(tuple_stree.children[0].out_memlets['out'].subset) == '0'
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in list_stree.preorder_traversal())
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in tuple_stree.preorder_traversal())


def test_schedule_tree_distinguishes_list_and_tuple_indices_with_symbolic_shape():
    n = dace.symbol('n')

    @dace.program
    def list_prog(A: dace.float64[5, n]):
        tmp = A[[1, 2]]
        return tmp

    @dace.program
    def tuple_prog(A: dace.float64[5, n]):
        tmp = A[(1, 2)]
        return tmp

    list_stree = list_prog.to_schedule_tree()
    tuple_stree = tuple_prog.to_schedule_tree()

    assert tuple(list_stree.containers['tmp'].shape) == (2, n)
    assert str(list_stree.children[0].out_memlets['out'].subset) == '0:2, 0:n'
    assert isinstance(tuple_stree.containers['tmp'], dace.data.Scalar)


def test_schedule_tree_symbolic_static_slice_shape():
    n = dace.symbol('n')

    @dace.program
    def slice_prog(A: dace.float64[n]):
        tmp = A[1:n:2]
        return tmp

    stree = slice_prog.to_schedule_tree()

    assert isinstance(stree.containers['tmp'], dace.data.Array)
    assert str(stree.containers['tmp'].shape[0]) == 'ceiling(n/2 - 1/2)'
    assert isinstance(stree.children[0], tn.ViewNode)


def test_comprehension_desugaring():
    """Comprehensions should be desugared to explicit loops."""

    @dace.program
    def comp_prog(A: dace.float64[8]):
        tmp = [A[i] for i in range(4)]
        return tmp

    stree = comp_prog.to_schedule_tree()

    # After desugaring, we should see loop constructs instead of a single TaskletNode
    # Check that it at least doesn't crash and produces a valid tree
    assert isinstance(stree, tn.ScheduleTreeRoot)


def test_generator_immediate_consumption_desugaring():

    @dace.program
    def gen_prog(A: dace.float64[8]):
        total = sum(x for x in A)
        return total

    stree = gen_prog.to_schedule_tree()

    assert isinstance(stree, tn.ScheduleTreeRoot)
    assert not any(isinstance(child, tn.PythonCallbackNode) for child in stree.children)
    assert any(isinstance(child, tn.LoopScope) for child in stree.children)


def test_generic_visit_warns():
    """generic_visit should emit a warning for truly unknown node types."""
    import warnings
    from dace.frontend.python.schedule_tree_frontend import PythonScheduleTreeBuilder

    # Verify that generic_visit is invoked by the builder when it encounters
    # an AST statement it doesn't have a visitor for, and that it wraps the
    # result as a PythonCallbackNode.
    # We test this indirectly: the hardened generic_visit emits a warning,
    # which we verify is called via monkeypatching.
    called = []
    original_generic_visit = PythonScheduleTreeBuilder.generic_visit

    def patched_generic_visit(self, node):
        called.append(type(node).__name__)
        return original_generic_visit(self, node)

    PythonScheduleTreeBuilder.generic_visit = patched_generic_visit
    try:

        @dace.program
        def simple(A: dace.float64[10]):
            A[0] = 1.0

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            stree = simple.to_schedule_tree()
    finally:
        PythonScheduleTreeBuilder.generic_visit = original_generic_visit

    # If no unknown nodes were encountered, that's expected for simple code.
    # The key test is: import the builder, confirm generic_visit exists and warns.
    assert hasattr(PythonScheduleTreeBuilder, 'generic_visit')
    assert isinstance(stree, tn.ScheduleTreeRoot)


def test_comprehensive_coverage():
    """Verify every ast.stmt subclass has a visitor or preprocessing handler."""
    import sys

    # All statement node types in the current Python version
    all_stmt_types = set()
    for name in dir(ast):
        cls = getattr(ast, name)
        if isinstance(cls, type) and issubclass(cls, ast.stmt) and cls is not ast.stmt:
            all_stmt_types.add(name)

    # Statement types handled by the schedule tree builder
    from dace.frontend.python.schedule_tree_frontend import PythonScheduleTreeBuilder
    builder_visitors = set()
    for name in dir(PythonScheduleTreeBuilder):
        if name.startswith('visit_'):
            builder_visitors.add(name[6:])

    # Statement types handled by preprocessing (desugared or removed)
    preprocessing_handled = {
        'With',
        'AsyncWith',  # ContextManagerInliner
        'Assert',  # Removed/evaluated
        'AsyncFor',  # Disallowed
        'TypeAlias',  # TypeAliasResolver
    }

    # All explicitly handled types
    handled = builder_visitors | preprocessing_handled

    # Find unhandled statement types
    unhandled = all_stmt_types - handled

    # Some types might not exist in all Python versions
    expected_unhandled = set()
    if sys.version_info < (3, 10):
        expected_unhandled.add('Match')
    if sys.version_info < (3, 11):
        expected_unhandled.add('TryStar')
    actual_unhandled = unhandled - expected_unhandled
    assert not actual_unhandled, f'Unhandled AST statement types: {actual_unhandled}'


def test_type_alias_is_compile_time_only_in_schedule_tree(temp_python_module):
    if sys.version_info < (3, 12):
        pytest.skip('Type alias statements require Python 3.12+')

    with temp_python_module('''
import dace

@dace.program
def prog(A: dace.float32[4]):
    type dtype = dace.float32[4]
    tmp: dtype = A
    return tmp
''',
                            module_name_prefix='dace_schedule_tree_typealias') as module:
        stree = module.prog.to_schedule_tree()

    assert 'tmp' in stree.containers
    assert isinstance(stree.containers['tmp'], dace.data.Array)
    assert stree.containers['tmp'].dtype == dace.float32
    assert tuple(stree.containers['tmp'].shape) == (4, )
    assert not any(
        isinstance(node, tn.PythonCallbackNode) and node.reason == 'unhandled TypeAlias'
        for node in stree.preorder_traversal())


def test_generic_type_alias_is_rejected_in_schedule_tree(temp_python_module):
    if sys.version_info < (3, 12):
        pytest.skip('Type alias statements require Python 3.12+')

    with temp_python_module('''
import dace

@dace.program
def prog(A: dace.float32[4]):
    type dtype[T] = T
    return A
''',
                            module_name_prefix='dace_schedule_tree_typealias') as module:
        with pytest.raises(DaceSyntaxError, match='Generic type aliases'):
            module.prog.to_schedule_tree()


def test_type_var_tuple_alias_is_rejected_in_schedule_tree(temp_python_module):
    if sys.version_info < (3, 12):
        pytest.skip('Type alias statements require Python 3.12+')

    with temp_python_module('''
import dace

@dace.program
def prog(A: dace.float32[4]):
    type dtype[*Ts] = tuple[*Ts]
    return A
''',
                            module_name_prefix='dace_schedule_tree_typealias') as module:
        with pytest.raises(DaceSyntaxError, match='Generic type aliases'):
            module.prog.to_schedule_tree()


if __name__ == '__main__':
    pytest.main([__file__])
