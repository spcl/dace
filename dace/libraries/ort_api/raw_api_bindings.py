import collections
import copy
import ctypes
import os
import re

from dace.codegen import codeobject, targets, compiler

from dace.libraries.onnx.environments import ONNXRuntime


class ORTAPIError(RuntimeError):
    """ Error thrown when an ORT function returns a non-zero Status. """
    pass


class keydefaultdict(collections.defaultdict):

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class OrtCUDAProviderOptions(ctypes.Structure):
    _fields_ = [
        ("device_id", ctypes.c_int),
        ("cudnn_conv_algo_search", ctypes.c_int),
        ("cuda_mem_limit", ctypes.c_size_t),
        ("do_copy_in_default_stream", ctypes.c_int),
        ("has_user_compute_stream", ctypes.c_int),
        ("user_compute_stream", ctypes.c_void_p),
    ]


class ORTCAPIInterface:
    dll = None

    # yapf: disable
    functions_to_expose = [
        "CreateEnv",
        "CreateSessionOptions",
        "CreateKernelSession",
        "CreateExecutableKernelContext",
        "ExecutableKernelContext_AddInput",
        "ExecutableKernelContext_AddOutput",
        "ExecutableKernelContext_AddAttributeString",
        "ExecutableKernelContext_AddAttributeStrings",
        "ExecutableKernelContext_AddAttributeFloat",
        "ExecutableKernelContext_AddAttributeFloats",
        "ExecutableKernelContext_AddAttributeInt",
        "ExecutableKernelContext_AddAttributeInts",
        "ExecutableKernelContext_AddAttributeTensor",
        "CreateExecutableKernel",
        "ExecutableKernel_IsOutputOnCpu",
        "ExecutableKernel_IsInputOnCpu",
        "ExecutableKernel_SetInput",
        "ExecutableKernel_SetOutput",
        "ExecutableKernel_Compute",
        "SessionOptionsAppendExecutionProvider_CUDA",
        "CreateCpuMemoryInfo",
        "CreateMemoryInfo",
        "CreateTensorWithDataAsOrtValue",
    ]
    release_functions_to_expose = [
        "ExecutableKernel",
        "ExecutableKernelContext",
        "KernelSession",
        "SessionOptions",
        "MemoryInfo",
        "Status",
        "Env",
        "Value"
    ]
    # yapf: enable
    enums_to_expose = {
        "OrtMemType": ["OrtMemTypeDefault", "OrtMemTypeCPU"],
        "OrtAllocatorType": ["OrtDeviceAllocator"],
        "OrtLoggingLevel": ["ORT_LOGGING_LEVEL_WARNING"],
        "ONNXTensorElementDataType": [
            "ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED", "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT",
            "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8", "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8",
            "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16", "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16",
            "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32", "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32",
            "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64", "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64",
            "ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING", "ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL",
            "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16", "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE",
            "ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64", "ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128",
            "ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16"
        ],
        "OrtCudnnConvAlgoSearch": ["EXHAUSTIVE", "HEURISTIC", "DEFAULT"]
    }

    _function_signatures = collections.defaultdict(list)

    def __enter__(self):
        if ORTCAPIInterface.dll is None:
            # build the dll
            ORTCAPIInterface.dll = self.build_dll()

        self.dll.GetErrorMessage.restype = ctypes.c_char_p

        # lazily constructed dict of function pointers
        self._function_pointers = keydefaultdict(lambda name: self._dll_get_fptr(name))
        self.exit_lambdas = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for lbd in self.exit_lambdas:
            lbd()

    def _dll_get_fptr(self, function_name):
        func_ptr = getattr(self.dll, function_name)
        if function_name in self.functions_to_expose:
            # annotate functions that return a status
            func_ptr.restype = ctypes.c_void_p
        return func_ptr

    def __getattr__(self, function_name):

        def wrapper(*args):
            if function_name not in ORTCAPIInterface._function_signatures:
                raise RuntimeError(
                    f"{function_name} was not exposed when this checker was compiled. To expose it, add it "
                    f"to ORTCAPIInterface.functions_to_expose")

            func_ptr = self._function_pointers[function_name]
            sig = self._function_signatures[function_name]
            if len(args) != len(sig):
                raise TypeError(f"OrtCAPIInterface.{function_name}() expected {len(sig)} arguments, got {len(args)}.")

            converted_args = []
            for arg_typ, arg in zip(sig, args):
                try:
                    converted_args.append(arg_typ(arg))
                except Exception as e:
                    raise TypeError(f"Could not convert argument {arg}") from e

            result = func_ptr(*converted_args)
            if not function_name.startswith("Release"):
                self._check_status(result)

        wrapper.__name__ = function_name
        return wrapper

    def _check_status(self, status: ctypes.c_void_p):
        if status:
            msg = self.dll.GetErrorMessage(ctypes.c_void_p(status))
            error_string = copy.deepcopy(msg.decode("ascii"))
            self.dll.ReleaseStatus(ctypes.c_void_p(status))
            raise ORTAPIError(error_string)

    def get_enum_value(self, enum_value_name: str) -> ctypes.c_int:
        """ Given an enum value, get the integer that represents it.

            :param enum_value_name: the string containing the value name.
            :return: the integer value.
        """
        return ctypes.c_int(self._function_pointers[f"Get{enum_value_name}"]())

    # Code generation methods
    def _get_function_def(self, header_code, function_name):
        # find the arguments of the function
        match = re.search(rf"ORT_API2_STATUS\({function_name},(.*?)\)", header_code)
        if match is None:
            raise RuntimeError(f"Couldn't parse ORT header file (couldn't find function {function_name} in header)")

        # remove annotations like _Inout_, _In_, etc.
        args_str = re.sub(r"_[^_\s]*?_", "", match[1].strip())
        # remove long whitespace
        args_str = re.sub(r"\s+", " ", args_str.strip())

        # figure out the names of the function parameters
        arg_names = []
        for arg in args_str.split(","):
            c_var_name_regex = r"[a-zA-Z_][a-zA-Z_0-9]*"
            match = re.search(fr"(?:const )?(?:enum )?\s*({c_var_name_regex})([*\s]+)({c_var_name_regex})", arg.strip())

            if match is None:
                raise RuntimeError(f"Couldn't parse ORT header file (couldn't parse argument '{arg}'"
                                   f" of function {function_name})")
            type_name, whitespace, var_name = match[1], match[2], match[3]
            arg_names.append(var_name)

            ctypes_mapping = {"int64_t": ctypes.c_int64}
            num_indirection = sum(1 if c == "*" else 0 for c in whitespace)
            if type_name in ctypes_mapping and num_indirection == 0:
                ctypes_type = ctypes_mapping[type_name]
            elif type_name == "char" and num_indirection == 1:
                ctypes_type = lambda x: ctypes.c_char_p(x.encode("ascii"))
            elif hasattr(ctypes, f"c_{type_name}") and num_indirection == 0:
                ctypes_type = getattr(ctypes, f"c_{type_name}")
            elif type_name in ORTCAPIInterface.enums_to_expose:
                ctypes_type = lambda x: self.get_enum_value(x)
            elif num_indirection > 0:
                # pointers must be converted manually
                ctypes_type = lambda x: x
            else:
                raise RuntimeError(f"Could not import function {function_name}: couldn't identify type {type_name}")

            ORTCAPIInterface._function_signatures[function_name].append(ctypes_type)

        return f"""
        extern "C" OrtStatus* {function_name}({args_str}) {{
            return ort_api->{function_name}({", ".join(arg_names)});
        }}
        """

    @staticmethod
    def _get_release_function(object_name) -> str:
        ORTCAPIInterface._function_signatures[f"Release{object_name}"] = [lambda x: x]
        return f"""
        extern "C" void Release{object_name}(Ort{object_name}* input) {{
            ort_api->Release{object_name}(input);
        }}
        """

    @staticmethod
    def _get_dtype_function(enum_name, enum_value_name) -> str:
        return f"""
        extern "C" {enum_name} Get{enum_value_name}(){{
            return {enum_value_name};
        }}
        """

    @staticmethod
    def _find_ort_header_path() -> str:
        from dace.libraries.onnx.environments.onnxruntime import ONNXRuntime
        for include_directory in ONNXRuntime.cmake_includes():
            header_path = os.path.join(include_directory, "onnxruntime_c_api.h")
            if os.path.exists(header_path):
                return header_path

        raise RuntimeError("Could not find onnxruntime_c_api.h")

    def _get_api_code(self) -> str:
        """ Generate the code for the ONNX C API header. We need to make our own header that allows us to call functions
            without indexing into the OrtApi struct. This allows us to link against these functions and call them from
            python.

            For example, for the function ReleaseStatus, our header needs to contain the following function that we can
            link against:

            void ReleaseStatus (OrtStatus* input) {
                ort_api->ReleaseStatus(input);
            }
        """

        header_path = ORTCAPIInterface._find_ort_header_path()
        with open(header_path, "r") as f:
            header_code = f.read()
        header_code = header_code.replace("\n", " ")
        return f"""\
        // AUTOGENERATED FILE: DO NOT EDIT
        #include <unordered_map>
        #include <string>
        #include <iostream>
        #include <sstream>
        #include "onnxruntime_c_api.h"
        #include "cpu_provider_factory.h"
        #include "cuda_provider_factory.h"

        // Start global ORT setup
        const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

        extern "C" const char* GetErrorMessage(const OrtStatus* status) {{
            return ort_api->GetErrorMessage(status);
        }}

        // exported API functions
        {"".join(ORTCAPIInterface._get_function_def(self, header_code, function_name) for function_name in ORTCAPIInterface.functions_to_expose)}

        // release functions
        {"".join(ORTCAPIInterface._get_release_function(object_name) for object_name in ORTCAPIInterface.release_functions_to_expose)}

        // dtype getters
        {"".join(ORTCAPIInterface._get_dtype_function(enum_name, enum_value_name)
                 for enum_name, values in ORTCAPIInterface.enums_to_expose.items()
                 for enum_value_name in values)}
        """

    def build_dll(self):
        program = codeobject.CodeObject("onnx_c_api_bridge",
                                        self._get_api_code(),
                                        "cpp",
                                        targets.cpu.CPUCodeGen,
                                        "ONNXCAPIBridge",
                                        environments={ONNXRuntime.full_class_path()})

        BUILD_PATH = os.path.join('.dacecache', "onnx_c_api_bridge")
        compiler.generate_program_folder(None, [program], BUILD_PATH)
        compiler.configure_and_compile(BUILD_PATH)

        api_dll = ctypes.CDLL(compiler.get_binary_name(BUILD_PATH, "onnx_c_api_bridge"))

        return api_dll
