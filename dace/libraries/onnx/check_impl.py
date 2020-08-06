import os
import inspect
import ctypes

import onnx

import dace
from dace.libraries.onnx.converters import dace_type_to_onnx_tensor_type
from dace.libraries.onnx.environments import has_cuda
from dace.codegen.codeobject import CodeObject
from dace.codegen.compiler import generate_program_folder, configure_and_compile, get_binary_name
from dace.codegen.targets import cpu


def build_checker():
    if hasattr(build_checker, "dll"):
        return build_checker.dll

    checker_code_path = os.path.join(
        os.path.dirname(inspect.getfile(dace.libraries.onnx)), "include",
        "op_checker.h")

    with open(checker_code_path, "r") as f:
        checker_code = f.read()

    program = CodeObject(
        "onnx_op_checker",
        checker_code.replace('// INSERT CUDA',
            "__ort_check_status(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, /*device=*/0));"
            if has_cuda() else ""),
        "cpp",
        cpu.CPUCodeGen,
        "ONNXOpChecker",
        environments={"ONNXRuntime"})

    BUILD_PATH = os.path.join('.dacecache', "onnx_op_checker")
    generate_program_folder(None, [program], BUILD_PATH)
    configure_and_compile(BUILD_PATH)

    checker_dll = ctypes.CDLL(get_binary_name(BUILD_PATH, "onnx_op_checker"))
    build_checker.dll = checker_dll

    return checker_dll


class OpChecker:
    def __init__(self, op_type: str, name: str):
        self.dll = build_checker()
        init_fn = self.dll.init_state
        init_fn.restype = ctypes.c_void_p
        # the cast is required for some reason
        self._state = ctypes.c_void_p(
            init_fn(op_type.encode("ascii"), name.encode("ascii")))

    def try_create(self):
        try_create = self.dll.try_create
        try_create.restype = ctypes.c_char_p
        result = try_create(self._state, 0)
        if result is not None:
            # raise an Error with the error message from ORT
            raise TypeError(result.decode("ascii"))

    def __del__(self, *args):
        if hasattr(self, "dll") and hasattr(self, "_state"):
            self.dll.free_state(self._state)

    def add_input(self, type: onnx.TensorProto.DataType):
        self.dll.add_input(self._state, type)

    def add_output(self, type: onnx.TensorProto.DataType):
        self.dll.add_output(self._state, type)

    def add_attribute(self, attribute: onnx.AttributeProto):
        self.dll.add_attribute()


def check_op(sdfg, state, node):
    """ Check whether a ONNXOp node has an implementation in ORT """
    checker = OpChecker(node.schema.name, node.name)
    for edge, is_input in node.iter_edges(state):
        edge_data = edge.data.data
        edge_dtype = sdfg.arrays[edge_data].dtype
        edge_dtype_onnx_type = dace_type_to_onnx_tensor_type(edge_dtype)
        if is_input:
            checker.add_input(edge_dtype_onnx_type)
        else:
            checker.add_output(edge_dtype_onnx_type)

    checker.try_create()
