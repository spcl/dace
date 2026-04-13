# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace import data, dtypes
from dace.frontend.python import preprocessing
from dace.frontend.python.schedule_tree import ScheduleTreeTypeInference
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _assert_pyobject_scalar(descriptor: data.Data) -> None:
    assert isinstance(descriptor, data.Scalar)
    assert descriptor.dtype == dtypes.pyobject()


def _infer_schedule_tree_bindings(program):
    modules = {name: value.__name__ for name, value in program.global_vars.items() if dtypes.ismodule(value)}
    modules['builtins'] = ''
    parsed_ast, _ = preprocessing.preprocess_dace_program(program.f, {'A': dace.float64[4]},
                                                          dict(program.global_vars),
                                                          modules,
                                                          resolve_functions=program.resolve_functions,
                                                          default_args=set(),
                                                          normalize_generic_for_loops=True,
                                                          preserve_object_attributes=True,
                                                          disallowed_stmts=set())
    return ScheduleTreeTypeInference(parsed_ast.program_globals, {
        'A': dace.float64[4]
    }).infer(parsed_ast.preprocessed_ast)


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
