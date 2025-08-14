import ctypes

import dace
import numpy as np

from dace.libraries.onnx.schema import ONNXAttributeType
from dace.libraries.onnx.converters import ONNX_DTYPES_TO_DACE_TYPE_CLASS
from dace.libraries.ort_api.raw_api_bindings import OrtCUDAProviderOptions, ORTCAPIInterface, ORTAPIError

dt_to_onnx_string = {v: k.upper() for k, v in ONNX_DTYPES_TO_DACE_TYPE_CLASS.items()}


class Env:

    def __init__(self, api):
        self.api = api

    def __enter__(self):
        self.ptr = ctypes.c_void_p()
        self.api.CreateEnv("ORT_LOGGING_LEVEL_WARNING", "ort_api", ctypes.byref(self.ptr))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api.ReleaseEnv(self.ptr)


class MemoryInfo:

    def __init__(self, api, cuda: bool, pinned: bool):
        self.cuda = cuda
        self.pinned = pinned
        self.api = api

    @staticmethod
    def for_cpu(api):
        return MemoryInfo(api, cuda=False, pinned=False)

    @staticmethod
    def for_cuda(api, pinned: bool):
        return MemoryInfo(api, cuda=True, pinned=pinned)

    def __enter__(self):
        self.ptr = ctypes.c_void_p()
        if not self.cuda:
            self.api.CreateCpuMemoryInfo("OrtDeviceAllocator", "OrtMemTypeDefault", ctypes.byref(self.ptr))
        else:
            self.api.CreateMemoryInfo(f"Cuda{'Pinned' if self.pinned else ''}", "OrtDeviceAllocator", 0,
                                      "OrtMemTypeCPU" if self.pinned else "OrtMemTypeDefault", ctypes.byref(self.ptr))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api.ReleaseMemoryInfo(self.ptr)


class SessionOptions:

    def __init__(self, api, cuda=False):
        self.api = api
        self.env = Env(api)
        self.cuda = cuda

    def __enter__(self):
        self.env.__enter__()
        self.ptr = ctypes.c_void_p()

        self.api.CreateSessionOptions(ctypes.byref(self.ptr))

        self.api.dll.OrtSessionOptionsAppendExecutionProvider_CPU(self.ptr, ctypes.c_int(0))

        if self.cuda and hasattr(self.api.dll, "OrtSessionOptionsAppendExecutionProvider_CUDA"):
            cuda_opts = OrtCUDAProviderOptions(device_id=0,
                                               cudnn_conv_algo_search=self.api.get_enum_value("DEFAULT"),
                                               cuda_mem_limit=np.iinfo(ctypes.c_size_t).max,
                                               do_copy_in_default_stream=1,
                                               has_user_compute_stream=0,
                                               user_compute_stream=0)

            self.api.SessionOptionsAppendExecutionProvider_CUDA(self.ptr, ctypes.byref(cuda_opts))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api.ReleaseSessionOptions(self.ptr)
        self.env.__exit__(exc_type, exc_val, exc_tb)


class KernelSession:
    """
    Transitional wrapper that used to create a KernelSession.
    Now it optionally creates a standard OrtSession (if env + model are given),
    or just manages SessionOptions with no session handle.
    """

    def __init__(self, api, cuda: bool = False, *, env=None, model_bytes: bytes = None, model_path: str = None):
        self.api = api
        self.session_options = SessionOptions(api, cuda=cuda)
        self.env = env
        self.model_bytes = model_bytes
        self.model_path = model_path
        self.ptr = None  # OrtSession* (if created)

    def __enter__(self):
        so_ptr = self.session_options.__enter__()  # prepares OrtSessionOptions*
        self.ptr = ctypes.c_void_p()

        # If env + model are supplied, create a normal OrtSession.
        if self.env is not None:
            if self.model_bytes:
                buf = (ctypes.c_char * len(self.model_bytes)).from_buffer_copy(self.model_bytes)
                self.api.CreateSessionFromArray(self.env, buf, ctypes.c_size_t(len(self.model_bytes)),
                                                so_ptr.ptr, ctypes.byref(self.ptr))
            elif self.model_path:
                # ORTCHAR_T* conversion is handled by your arg mapper
                self.api.CreateSession(self.env, self.model_path, so_ptr.ptr, ctypes.byref(self.ptr))
            else:
                # No model supplied: leave self.ptr as NULL; this is fine for code paths
                # that only needed the old KernelSession side-effects.
                pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If we created a real OrtSession, release it.
        if isinstance(self.ptr, ctypes.c_void_p) and self.ptr.value:
            self.api.ReleaseSession(self.ptr)
        self.session_options.__exit__(exc_type, exc_val, exc_tb)


class ExecutableKernelContext:

    def __init__(self, api: ORTCAPIInterface, kernel_session: KernelSession, name, op_type):
        self.kernel_session = kernel_session
        self.api = api
        self.n_inputs = 0
        self.n_outputs = 0
        self.name = name
        self.op_type = op_type

    def __enter__(self):
        self.ptr = ctypes.c_void_p()
        self.api.CreateExecutableKernelContext(self.name, self.op_type, ctypes.byref(self.ptr))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api.ReleaseExecutableKernelContext(self.ptr)

    def add_input(self, dtype: dace.typeclass):
        self.n_inputs += 1
        self.api.ExecutableKernelContext_AddInput(self.ptr, f"ONNX_TENSOR_ELEMENT_DATA_TYPE_{dt_to_onnx_string[dtype]}")

    def add_output(self, dtype: dace.typeclass):
        self.n_outputs += 1
        self.api.ExecutableKernelContext_AddOutput(self.ptr,
                                                   f"ONNX_TENSOR_ELEMENT_DATA_TYPE_{dt_to_onnx_string[dtype]}")

    def add_attribute(self, attr_name, attr_value, attr_type: ONNXAttributeType):
        if attr_value is None:
            return
        attr_name = attr_name
        add_attr_function = getattr(self.api, f"ExecutableKernelContext_AddAttribute{attr_type.name}")

        if attr_type == ONNXAttributeType.Int or attr_type == ONNXAttributeType.Float or attr_type == ONNXAttributeType.String:
            add_attr_function(self.ptr, attr_name, attr_value)
        elif attr_type == ONNXAttributeType.Ints or attr_type == ONNXAttributeType.Floats or attr_type == ONNXAttributeType.Strings:
            get_elem_ctype = {
                ONNXAttributeType.Ints: ctypes.c_int64,
                ONNXAttributeType.Floats: ctypes.c_float,
                ONNXAttributeType.Strings: ctypes.c_char_p
            }
            elem_ctype = get_elem_ctype[attr_type]
            array_type = elem_ctype * len(attr_value)
            data_p = array_type(*attr_value)
            add_attr_function(self.ptr, attr_name, data_p, len(attr_value))
        elif attr_type == ONNXAttributeType.Tensor:

            data = [data_val.item() for data_val in np.nditer(attr_value)]
            ctype = np.ctypeslib.as_ctypes_type(attr_value.dtype)
            type_str = dt_to_onnx_string[dace.DTYPE_TO_TYPECLASS[attr_value.dtype.type]]
            type = f"ONNX_TENSOR_ELEMENT_DATA_TYPE_{type_str}"
            p_data = (ctype * len(data))(*data)
            p_data = ctypes.cast(p_data, ctypes.c_void_p)
            shape = (ctypes.c_int64 * len(attr_value.shape))(*attr_value.shape)
            add_attr_function(self.ptr, attr_name, p_data, len(data), shape, len(attr_value.shape), type)

    def try_create_kernel(self, provider_id: int) -> "ExecutableKernel":
        return ExecutableKernel(self.api, self, provider_id)


class Value:

    def __init__(self, api, array: np.ndarray):
        self.api = api
        self.mem_info = MemoryInfo.for_cpu(api)
        self.array = array

    def __enter__(self):
        self.ptr = ctypes.c_void_p()

        shape = (ctypes.c_int64 * len(self.array.shape))(*self.array.shape)
        data_ctype = np.ctypeslib.as_ctypes_type(self.array.dtype)

        data_ptr = ctypes.c_void_p(self.array.__array_interface__['data'][0])

        type_str = dt_to_onnx_string[dace.DTYPE_TO_TYPECLASS[self.array.dtype.type]]
        type = f"ONNX_TENSOR_ELEMENT_DATA_TYPE_{type_str}"

        with self.mem_info:
            self.api.CreateTensorWithDataAsOrtValue(self.mem_info.ptr, data_ptr,
                                                    ctypes.sizeof(data_ctype) * self.array.size, shape,
                                                    len(self.array.shape), type, ctypes.byref(self.ptr))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api.ReleaseValue(self.ptr)


class ExecutableKernel:

    def __init__(self, api, kernel_context: ExecutableKernelContext, provider_id: int):
        self.api = api
        self.provider_id = provider_id
        self.kernel_context = kernel_context
        self.values = []

    def __enter__(self):
        self.ptr = ctypes.c_void_p()
        self.api.CreateExecutableKernel(self.kernel_context.kernel_session.ptr, self.kernel_context.ptr,
                                        self.provider_id, ctypes.byref(self.ptr))
        return self

    def add_input(self, array: np.ndarray, idx: int):
        value = Value(self.api, array).__enter__()
        self.values.append(value)
        self.api.ExecutableKernel_SetInput(self.ptr, idx, value.ptr)

    def add_output(self, array: np.ndarray, idx: int):
        value = Value(self.api, array).__enter__()
        self.values.append(value)
        self.api.ExecutableKernel_SetOutput(self.ptr, idx, value.ptr)

    def compute(self):
        self.api.ExecutableKernel_Compute(self.ptr)

    def check_io_locations(self):

        outputs_on_cpu = []
        inputs_on_cpu = []

        for i in range(self.kernel_context.n_outputs):
            result = ctypes.c_int(-1)
            self.api.ExecutableKernel_IsOutputOnCpu(self.ptr, i, ctypes.byref(result))
            if result == -1:
                raise ORTAPIError("Could not determine output storage of op")
            outputs_on_cpu.append(bool(result))

        for i in range(self.kernel_context.n_inputs):
            result = ctypes.c_int(-1)
            self.api.ExecutableKernel_IsInputOnCpu(self.ptr, i, ctypes.byref(result))
            if result == -1:
                raise ORTAPIError("Could not determine output storage of op")
            inputs_on_cpu.append(bool(result))

        return inputs_on_cpu, outputs_on_cpu

    def __exit__(self, exc_type, exc_val, exc_tb):

        for value in self.values:
            self.api.ReleaseValue(value.ptr)

        self.api.ReleaseExecutableKernel(self.ptr)
