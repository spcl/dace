# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast
import sys

import dace
import pytest
from dace import data, dtypes
from dace.frontend.python import preprocessing
from dace.frontend.python.schedule_tree import ScheduleTreeTypeInference
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _assert_pyobject_scalar(descriptor: data.Data) -> None:
    assert isinstance(descriptor, data.Scalar)
    assert descriptor.dtype == dtypes.pyobject()


def _assert_string_scalar(descriptor: data.Data) -> None:
    assert isinstance(descriptor, data.Scalar)
    assert descriptor.dtype == dtypes.string


def _infer_schedule_tree_bindings(program, argtypes=None):
    argtypes = dict(argtypes or {'A': dace.float64[4]})
    modules = {name: value.__name__ for name, value in program.global_vars.items() if dtypes.ismodule(value)}
    modules['builtins'] = ''
    parsed_ast, _ = preprocessing.preprocess_dace_program(program.f,
                                                          argtypes,
                                                          dict(program.global_vars),
                                                          modules,
                                                          resolve_functions=program.resolve_functions,
                                                          default_args=set(),
                                                          normalize_generic_for_loops=True,
                                                          preserve_object_attributes=True,
                                                          disallowed_stmts=set())
    return ScheduleTreeTypeInference(parsed_ast.program_globals, argtypes).infer(parsed_ast.preprocessed_ast)


def test_schedule_tree_type_inference_opaque_assignments_use_pyobject_scalar():

    class Box:

        def __init__(self):
            self.value = None

    @dace.program
    def prog(A: dace.float64[4]):
        box = Box()
        tmp = box.value
        A[0] = A[0]

    bindings = _infer_schedule_tree_bindings(prog)

    _assert_pyobject_scalar(bindings['box'].descriptor)
    _assert_pyobject_scalar(bindings['tmp'].descriptor)


def test_python_frontend_schedule_tree_opaque_assignments_use_pyobject_scalar():

    class Box:

        def __init__(self):
            self.value = None

    @dace.program
    def prog(A: dace.float64[4]):
        box = Box()
        tmp = box.value
        A[0] = A[0]

    stree = prog.to_schedule_tree()

    _assert_pyobject_scalar(stree.containers['box'])
    _assert_pyobject_scalar(stree.containers['tmp'])
    assert isinstance(stree.children[0], tn.AssignNode)
    assert isinstance(stree.children[1], tn.TaskletNode)


def test_python_frontend_schedule_tree_callback_outputs_use_pyobject_scalar():

    @dace.program
    def prog(A: dace.float64[4]):
        import math as m
        tmp = m
        A[0] = A[0]

    stree = prog.to_schedule_tree()

    assert isinstance(stree.children[0], tn.PythonCallbackNode)
    _assert_pyobject_scalar(stree.containers['m'])
    _assert_pyobject_scalar(stree.containers['tmp'])


def test_schedule_tree_type_inference_dict_same_key_update_widens_value_type():

    @dace.program
    def prog(A: dace.float64[2]):
        mapping = {'left': A[0], 'right': A[1]}
        mapping['left'] = 'two'
        value = mapping['left']
        return 0.0

    bindings = _infer_schedule_tree_bindings(prog, {'A': dace.float64[2]})

    assert bindings['mapping'].descriptor.value_type.dtype == dtypes.pyobject()
    _assert_string_scalar(bindings['value'].descriptor)


def test_schedule_tree_type_inference_dict_known_static_reads_stay_precise():

    @dace.program
    def prog(A: dace.float64[2]):
        mapping = {'left': A[0], 'right': 'two'}
        left = mapping['left']
        right = mapping['right']
        return 0.0

    bindings = _infer_schedule_tree_bindings(prog, {'A': dace.float64[2]})

    assert bindings['mapping'].descriptor.value_type.dtype == dtypes.pyobject()
    assert isinstance(bindings['left'].descriptor, data.Scalar)
    assert bindings['left'].descriptor.dtype == dace.float64
    _assert_string_scalar(bindings['right'].descriptor)


def test_schedule_tree_type_inference_constant_scalars_use_literal_descriptors():

    @dace.program
    def prog(A: dace.float64[4]):
        text = 'bla'
        number = 5.03
        A[0] = A[0]

    bindings = _infer_schedule_tree_bindings(prog)

    _assert_string_scalar(bindings['text'].descriptor)
    assert isinstance(bindings['number'].descriptor, data.Scalar)
    assert bindings['number'].descriptor.dtype == dace.float64


def test_python_frontend_schedule_tree_constant_scalars_use_literal_descriptors():

    @dace.program
    def prog(A: dace.float64[4]):
        text = 'bla'
        number = 5.03
        A[0] = A[0]

    stree = prog.to_schedule_tree()

    _assert_string_scalar(stree.containers['text'])
    assert isinstance(stree.containers['number'], data.Scalar)
    assert stree.containers['number'].dtype == dace.float64


def test_python_frontend_schedule_tree_runtime_fstring_callback_outputs_use_string_scalar():

    @dace.program
    def prog(i: dace.int32):
        return f'value={i}'

    stree = prog.to_schedule_tree()

    callbacks = [node for node in stree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]

    assert len(callbacks) == 1
    assert callbacks[0].reason == 'f-string'
    assert len(callbacks[0].output_names) == 1
    result_name = callbacks[0].output_names[0]
    _assert_string_scalar(stree.containers[result_name])
    assert isinstance(stree.children[-1], tn.ReturnNode)
    assert stree.children[-1].values[0] == result_name


def test_schedule_tree_type_inference_nested_generic_conflicts_do_not_leak():
    if sys.version_info < (3, 12):
        pytest.skip('Generic function type parameters require Python 3.12+')

    source = '''
def prog[T, *Ts](A):
    tmp = A

    def inner[T, *Ts](x: T, y: tuple[*Ts]):
        tmp = 1
        return x

    return tmp
'''

    module = ast.parse(source)
    function = module.body[0]
    bindings = ScheduleTreeTypeInference({'dace': dace}, {'A': dace.float64[4]}).infer(function)

    assert 'tmp' in bindings
    assert isinstance(bindings['tmp'].descriptor, data.Array)
    assert bindings['tmp'].descriptor.dtype == dace.float64
    assert tuple(bindings['tmp'].descriptor.shape) == (4, )


def test_schedule_tree_type_inference_distinguishes_list_and_tuple_indices():

    @dace.program
    def list_prog(A: dace.float64[5, 6]):
        tmp = A[[1, 2]]

    @dace.program
    def tuple_prog(A: dace.float64[5, 6]):
        tmp = A[(1, 2)]

    list_bindings = _infer_schedule_tree_bindings(list_prog, {'A': dace.float64[5, 6]})
    tuple_bindings = _infer_schedule_tree_bindings(tuple_prog, {'A': dace.float64[5, 6]})

    assert isinstance(list_bindings['tmp'].descriptor, data.Array)
    assert tuple(list_bindings['tmp'].descriptor.shape) == (2, 6)
    assert list_bindings['tmp'].descriptor.dtype == dace.float64
    assert isinstance(tuple_bindings['tmp'].descriptor, data.Scalar)
    assert tuple_bindings['tmp'].descriptor.dtype == dace.float64
    assert tuple_bindings['tmp'].kind == 'scalar'


def test_schedule_tree_type_inference_distinguishes_list_and_tuple_indices_with_symbolic_shape():
    n = dace.symbol('n')

    @dace.program
    def list_prog(A: dace.float64[5, n]):
        tmp = A[[1, 2]]

    @dace.program
    def tuple_prog(A: dace.float64[5, n]):
        tmp = A[(1, 2)]

    list_bindings = _infer_schedule_tree_bindings(list_prog, {'A': dace.float64[5, n]})
    tuple_bindings = _infer_schedule_tree_bindings(tuple_prog, {'A': dace.float64[5, n]})

    assert tuple(list_bindings['tmp'].descriptor.shape) == (2, n)
    assert isinstance(tuple_bindings['tmp'].descriptor, data.Scalar)


def test_schedule_tree_type_inference_symbolic_static_slice_shape():
    n = dace.symbol('n')

    @dace.program
    def slice_prog(A: dace.float64[n]):
        tmp = A[1:n:2]

    bindings = _infer_schedule_tree_bindings(slice_prog, {'A': dace.float64[n]})

    assert isinstance(bindings['tmp'].descriptor, data.Array)
    assert str(bindings['tmp'].descriptor.shape[0]) == 'ceiling(n/2 - 1/2)'


def test_schedule_tree_type_inference_ellipsis_shape():
    n = dace.symbol('n')

    @dace.program
    def ellipsis_prog(A: dace.float64[4, n, 6, 7]):
        tmp = A[1:3, ..., 0]

    bindings = _infer_schedule_tree_bindings(ellipsis_prog, {'A': dace.float64[4, n, 6, 7]})

    assert isinstance(bindings['tmp'].descriptor, data.Array)
    assert bindings['tmp'].descriptor.dtype == dace.float64
    assert tuple(bindings['tmp'].descriptor.shape) == (2, n, 6)
