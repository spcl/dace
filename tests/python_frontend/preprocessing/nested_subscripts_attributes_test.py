# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the preprocessing functionality of unnesting nested subscripts and attributes. """

import ast
import inspect
import numpy as np

from dace.frontend.python.astutils import _remove_outer_indentation
from dace.frontend.python import preprocessing as pr
from numpy import typing as npt


def test_attribute_on_op():

    def original_function(A: npt.NDArray[np.int32], B: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        return (A + B).T
    
    function_ast = ast.parse(_remove_outer_indentation(inspect.getsource(original_function)))

    name_getter = pr.NameGetter()
    name_getter.visit(function_ast)
    program_names = name_getter.names

    function_ast = pr.ParentSetter().visit(function_ast)
    subatrr_replacer = pr.AttributeTransformer(names=program_names)
    function_ast = subatrr_replacer.visit(function_ast)
    for parent, attr, idx, node in reversed(subatrr_replacer.ast_nodes_to_add):
        getattr(parent, attr).insert(idx, node)
    
    ast.fix_missing_locations(function_ast)

    name_getter_2 = pr.NameGetter()
    name_getter_2.visit(function_ast)
    program_names_2 = name_getter_2.names

    for i in range(2):
        name = f'__var_{i}'
        assert name in program_names_2
    
    code = compile(function_ast, filename='<ast>', mode='exec')
    print(ast.unparse(function_ast))


def test_attribute_call():

    def original_function(A: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        return A.sum(axis=0)
    
    function_ast = ast.parse(_remove_outer_indentation(inspect.getsource(original_function)))

    name_getter = pr.NameGetter()
    name_getter.visit(function_ast)
    program_names = name_getter.names

    function_ast = pr.ParentSetter().visit(function_ast)
    subatrr_replacer = pr.AttributeTransformer(names=program_names)
    function_ast = subatrr_replacer.visit(function_ast)
    for parent, attr, idx, node in reversed(subatrr_replacer.ast_nodes_to_add):
        getattr(parent, attr).insert(idx, node)
    
    ast.fix_missing_locations(function_ast)

    name_getter_2 = pr.NameGetter()
    name_getter_2.visit(function_ast)
    program_names_2 = name_getter_2.names

    for i in range(1):
        name = f'__var_{i}'
        assert name in program_names_2
    
    code = compile(function_ast, filename='<ast>', mode='exec')
    print(ast.unparse(function_ast))


def test_attribute_call_on_op():

    def original_function(A: npt.NDArray[np.int32], B: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        return (A + B).sum(axis=0)
    
    function_ast = ast.parse(_remove_outer_indentation(inspect.getsource(original_function)))

    name_getter = pr.NameGetter()
    name_getter.visit(function_ast)
    program_names = name_getter.names

    function_ast = pr.ParentSetter().visit(function_ast)
    subatrr_replacer = pr.AttributeTransformer(names=program_names)
    function_ast = subatrr_replacer.visit(function_ast)
    for parent, attr, idx, node in reversed(subatrr_replacer.ast_nodes_to_add):
        getattr(parent, attr).insert(idx, node)
    
    ast.fix_missing_locations(function_ast)

    name_getter_2 = pr.NameGetter()
    name_getter_2.visit(function_ast)
    program_names_2 = name_getter_2.names

    for i in range(1):
        name = f'__var_{i}'
        assert name in program_names_2
    
    code = compile(function_ast, filename='<ast>', mode='exec')
    print(ast.unparse(function_ast))


def original_function(A: npt.NDArray[np.int32],
                      i0: npt.NDArray[np.int32],
                      i1: npt.NDArray[np.int32],
                      i2: npt.NDArray[np.int32],
                      i3: npt.NDArray[np.int32],
                      i4: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    
    B = np.zeros_like(A)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                for l in range(A.shape[3]):
                    B[i, :max(i2[j], i3[j]) - min(i2[j], i3[j]), k, l] = (
                        A[i0[i1[i]], min(i2[j], i3[j]) : max(i2[j], i3[j]), k, i4[l]])

    return (A + B).sum(axis=0)


def test_0():
    
    function_ast = ast.parse(inspect.getsource(original_function))

    name_getter = pr.NameGetter()
    name_getter.visit(function_ast)
    program_names = name_getter.names

    function_ast = pr.ParentSetter().visit(function_ast)
    subatrr_replacer = pr.NestedSubsAttrsReplacer(names=program_names)
    function_ast = subatrr_replacer.visit(function_ast)
    for parent, idx, node in reversed(subatrr_replacer.ast_nodes_to_add):
        parent.body.insert(idx, node)
    
    ast.fix_missing_locations(function_ast)

    name_getter_2 = pr.NameGetter()
    name_getter_2.visit(function_ast)
    program_names_2 = name_getter_2.names

    for i in range(7):
        name = f'__var_{i}'
        assert name in program_names_2
    
    code = compile(function_ast, filename='<ast>', mode='exec')
    namespace = {**globals()}
    exec(code, namespace)
    new_function = namespace['original_function']

    rng = np.random.default_rng(42)

    A = rng.integers(0, 100, size=(5, 5, 5, 5), dtype=np.int32)
    i0 = rng.integers(0, 5, size=(5,), dtype=np.int32)
    i1 = rng.integers(0, 5, size=(5,), dtype=np.int32)
    i2 = rng.integers(0, 5, size=(5,), dtype=np.int32)
    i3 = rng.integers(0, 5, size=(5,), dtype=np.int32)
    i4 = rng.integers(0, 5, size=(5,), dtype=np.int32)

    ref = original_function(A, i0, i1, i2, i3, i4)
    val = new_function(A, i0, i1, i2, i3, i4)

    assert np.allclose(ref, val)


if __name__ == '__main__':
    # test_0()
    test_attribute_on_op()
    test_attribute_call()
    test_attribute_call_on_op()
