import os
import inspect
import ctypes
from typing import Optional, List, Tuple

import numpy as np

import dace
from dace.dtypes import DTYPE_TO_TYPECLASS
from dace.libraries.onnx.schema import ONNXAttributeType
from dace.libraries.onnx.converters import ONNX_DTYPES_TO_DACE_TYPE_CLASS
from dace.codegen.codeobject import CodeObject
from dace.codegen.compiler import generate_program_folder, configure_and_compile, get_binary_name
from dace.codegen.targets import cpu


class ONNXOpValidationError(Exception):
    pass


def build_checker():
    if hasattr(build_checker, "dll"):
        return build_checker.dll

    checker_code_path = os.path.join(
        os.path.dirname(inspect.getfile(dace.libraries.onnx)), "include",
        "op_checker.h")

    with open(checker_code_path, "r") as f:
        checker_code = f.read()

    program = CodeObject("onnx_op_checker",
                         checker_code,
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
    def __init__(self, op_type: str, name: str, check_io_locations=False):

        self.n_outputs = 0
        self.n_inputs = 0
        self.check_io_locations = check_io_locations
        self.name = name.encode("ascii")
        self.op_type = op_type.encode("ascii")
        self.dll = build_checker()


    def __enter__(self):
        self._ReleaseStatus = self._get_function("ReleaseStatus")
        self._ReleaseEnv = self._get_function("ReleaseEnv")
        self._ReleaseSessionOptions = self._get_function(
            "ReleaseSessionOptions")
        self._ReleaseKernelSession = self._get_function("ReleaseKernelSession")
        self._ReleaseExecutableKernel = self._get_function(
            "ReleaseExecutableKernel")
        self._ReleaseExecutableKernelContext = self._get_function(
            "ReleaseExecutableKernelContext")

        self._GetErrorMessage = self._get_function("GetErrorMessage",
                                                   restype=ctypes.c_char_p)
        self._env = ctypes.c_void_p()
        _CreateEnv = self._get_function("CreateEnv", restype=ctypes.c_void_p)
        self._check_status(_CreateEnv(ctypes.byref(self._env)))

        self._session_options = ctypes.c_void_p()
        _CreateSessionOptions = self._get_function("CreateSessionOptions",
                                                   restype=ctypes.c_void_p)
        self._check_status(
            _CreateSessionOptions(ctypes.byref(self._session_options)))

        append_cpu = self._get_function(
            "OrtSessionOptionsAppendExecutionProvider_CPU",
            restype=ctypes.c_void_p)
        self._check_status(append_cpu(self._session_options, ctypes.c_int(0)))
        if hasattr(self.dll, "OrtSessionOptionsAppendExecutionProvider_CUDA"):
            append_cuda = self._get_function(
                "OrtSessionOptionsAppendExecutionProvider_CUDA",
                restype=ctypes.c_void_p)
            self._check_status(
                append_cuda(self._session_options, ctypes.c_int(0)))

        self._session = ctypes.c_void_p()
        _CreateKernelSession = self._get_function("CreateKernelSession",
                                                  restype=ctypes.c_void_p)
        self._check_status(
            _CreateKernelSession(self._session_options,
                                 ctypes.byref(self._session)))

        self._context = ctypes.c_void_p()
        _CreateExecutableKernelContext = self._get_function(
            "CreateExecutableKernelContext", restype=ctypes.c_void_p)
        self._check_status(
            _CreateExecutableKernelContext(ctypes.c_char_p(self.name),
                                           ctypes.c_char_p(self.op_type),
                                           ctypes.byref(self._context)))

        self._CreateExecutableKernel = self._get_function(
            "CreateExecutableKernel", restype=ctypes.c_void_p)
        self._AddInput = self._get_function("ExecutableKernelContext_AddInput",
                                            restype=ctypes.c_void_p)
        self._AddOutput = self._get_function(
            "ExecutableKernelContext_AddOutput", restype=ctypes.c_void_p)
        self._ExecutableKernelContext_IsOutputOnCpu = self._get_function(
            "ExecutableKernel_IsOutputOnCpu", restype=ctypes.c_void_p)
        self._ExecutableKernelContext_IsInputOnCpu = self._get_function(
            "ExecutableKernel_IsInputOnCpu", restype=ctypes.c_void_p)

        self._AddAttribute = {
            ONNXAttributeType.Int:
            self._get_function("ExecutableKernelContext_AddAttributeInt",
                               restype=ctypes.c_void_p),
            ONNXAttributeType.Ints:
            self._get_function("ExecutableKernelContext_AddAttributeInts",
                               restype=ctypes.c_void_p),
            ONNXAttributeType.Float:
            self._get_function("ExecutableKernelContext_AddAttributeFloat",
                               restype=ctypes.c_void_p),
            ONNXAttributeType.Floats:
            self._get_function("ExecutableKernelContext_AddAttributeFloats",
                               restype=ctypes.c_void_p),
            ONNXAttributeType.String:
            self._get_function("ExecutableKernelContext_AddAttributeString",
                               restype=ctypes.c_void_p),
            ONNXAttributeType.Strings:
            self._get_function("ExecutableKernelContext_AddAttributeStrings",
                               restype=ctypes.c_void_p),
            ONNXAttributeType.Tensor:
            self._get_function("ExecutableKernelContext_AddAttributeTensor",
                               restype=ctypes.c_void_p),
        }
        self.dt_to_onnx_string = {
            v: k.upper()
            for k, v in ONNX_DTYPES_TO_DACE_TYPE_CLASS.items()
        }
        return self

    def try_create(self, cuda=False) -> Optional[Tuple[List[bool], List[bool]]]:
        kernel = ctypes.c_void_p()
        self._check_status(
            self._CreateExecutableKernel(self._session, self._context,
                                         ctypes.c_size_t(1 if cuda else 0),
                                         ctypes.byref(kernel)))

        if self.check_io_locations:
            outputs_on_cpu = []
            inputs_on_cpu = []
            for i in range(self.n_outputs):
                result = ctypes.c_int(-1)
                self._ExecutableKernelContext_IsOutputOnCpu(kernel, ctypes.c_int(i), ctypes.byref(result))
                if result == -1:
                    raise ONNXOpValidationError("Could not determine output storage of op")
                outputs_on_cpu.append(bool(result))

            for i in range(self.n_inputs):
                result = ctypes.c_int(-1)
                self._ExecutableKernelContext_IsInputOnCpu(kernel, ctypes.c_int(i), ctypes.byref(result))
                if result == -1:
                    raise ONNXOpValidationError("Could not determine output storage of op")
                inputs_on_cpu.append(bool(result))

            self._ReleaseExecutableKernel(kernel)
            return inputs_on_cpu, outputs_on_cpu
        else:
            self._ReleaseExecutableKernel(kernel)

    def _get_function(self, symbol_name, restype=None):
        func = getattr(self.dll, symbol_name)
        func.restype = restype
        if restype is None:
            return func
        else:
            return lambda *args: restype(func(*args))

    def _check_status(self, status: ctypes.c_void_p):
        if status:
            error = self._GetErrorMessage(status)
            self._ReleaseStatus(status)
            print(error.value.decode("ascii"))
            raise ONNXOpValidationError("see error printed above")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not hasattr(self, "dll"):
            return

        if hasattr(self, "_context") and self._context:
            self._ReleaseExecutableKernelContext(self._context)

        if hasattr(self, "_session") and self._session:
            self._ReleaseKernelSession(self._session)

        if hasattr(self, "_session_options") and self._session_options:
            self._ReleaseSessionOptions(self._session_options)

        if hasattr(self, "_env") and self._env:
            self._ReleaseEnv(self._env)

    def add_input(self, dtype: dace.typeclass):
        self.n_inputs += 1
        type = ctypes.c_int(
            getattr(
                self.dll, "GetONNX_TENSOR_ELEMENT_DATA_TYPE_{}".format(
                    self.dt_to_onnx_string[dtype]))())
        self._AddInput(self._context, type)

    def add_output(self, dtype: dace.typeclass):
        self.n_outputs += 1
        type = ctypes.c_int(
            getattr(
                self.dll, "GetONNX_TENSOR_ELEMENT_DATA_TYPE_{}".format(
                    self.dt_to_onnx_string[dtype]))())
        self._AddOutput(self._context, type)

    def add_attribute(self, attr_name, attr_value,
                      attr_type: ONNXAttributeType):
        if attr_value is None:
            return
        attr_name = attr_name.encode("ascii")
        add_attr_function = self._AddAttribute[attr_type]
        if attr_type == ONNXAttributeType.Int or attr_type == ONNXAttributeType.Float or attr_type == ONNXAttributeType.String:
            to_ctype_value = {
                ONNXAttributeType.Int:
                ctypes.c_int64,
                ONNXAttributeType.Float:
                ctypes.c_float,
                ONNXAttributeType.String:
                lambda x: ctypes.c_char_p(x.encode("ascii"))
            }
            self._check_status(
                add_attr_function(self._context, ctypes.c_char_p(attr_name),
                                  to_ctype_value[attr_type](attr_value)))
        elif attr_type == ONNXAttributeType.Ints or attr_type == ONNXAttributeType.Floats or attr_type == ONNXAttributeType.Strings:
            get_elem_ctype = {
                ONNXAttributeType.Ints: ctypes.c_int64,
                ONNXAttributeType.Floats: ctypes.c_float,
                ONNXAttributeType.Strings: ctypes.c_char_p
            }
            elem_ctype = get_elem_ctype[attr_type]
            array_type = elem_ctype * len(attr_value)
            data_p = array_type(*attr_value)
            self._check_status(
                add_attr_function(self._context, ctypes.c_char_p(attr_name),
                                  data_p, ctypes.c_size_t(len(attr_value))))
        elif attr_type == ONNXAttributeType.Tensor:

            data = [data_val.item() for data_val in np.nditer(attr_value)]
            ctype = np.ctypeslib.as_ctypes_type(attr_value.dtype)
            type_str = self.dt_to_onnx_string[DTYPE_TO_TYPECLASS[
                attr_value.dtype.type]]
            type = ctypes.c_int(
                getattr(
                    self.dll,
                    "GetONNX_TENSOR_ELEMENT_DATA_TYPE_{}".format(type_str))())
            p_data = (ctype * len(data))(*data)
            p_data = ctypes.cast(p_data, ctypes.c_void_p)
            shape = (ctypes.c_int64 * len(attr_value.shape))(*attr_value.shape)
            self._check_status(
                add_attr_function(self._context, ctypes.c_char_p(attr_name),
                                  p_data, ctypes.c_size_t(len(data)), shape,
                                  ctypes.c_size_t(len(attr_value.shape)), type))


def check_op(sdfg, state, node, cuda=False) -> Tuple[List[bool], List[bool]]:
    """ Check whether a ONNXOp node has an implementation in ORT """
    with OpChecker(node.schema.name, node.name, check_io_locations=True) as checker:
        for attribute, onnx_attribute in node.schema.attributes.items():
            if hasattr(node, attribute):
                checker.add_attribute(attribute, getattr(node, attribute),
                                      onnx_attribute.type)

        for edge, is_input in node.iter_edges(state):
            edge_data = edge.data.data
            edge_dtype = sdfg.arrays[edge_data].dtype
            if is_input:
                checker.add_input(edge_dtype)
            else:
                checker.add_output(edge_dtype)

        return checker.try_create(cuda=cuda)
