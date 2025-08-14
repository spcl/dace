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
        # Core
        "CreateEnv",
        "CreateSessionOptions",
        "CreateSession",                 
        "CreateSessionFromArray",        
        "UpdateCUDAProviderOptions",
        "SessionOptionsAppendExecutionProvider_CUDA",      
        "SessionOptionsAppendExecutionProvider_CUDA_V2",   
        "CreateCUDAProviderOptions",                        
        "GetCUDAProviderOptionsAsString",                   
        # Memory / Tensors
        "CreateCpuMemoryInfo",
        "CreateMemoryInfo",
        "CreateTensorWithDataAsOrtValue",
    ]

    release_functions_to_expose = [
        "Session",          
        "SessionOptions",
        "CUDAProviderOptions",  
        "MemoryInfo",
        "Status",
        "Env",
        "Value",
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
        "OrtCudnnConvAlgoSearch": ["OrtCudnnConvAlgoSearchExhaustive", "OrtCudnnConvAlgoSearchHeuristic", "OrtCudnnConvAlgoSearchDefault"]
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
                    converted_args.append(self._coerce_ctypes_arg(arg_typ, arg))
                except Exception as e:
                    raise TypeError(f"Could not convert argument {arg}") from e

            result = func_ptr(*converted_args)
            if not function_name.startswith("Release"):
                self._check_status(result)

        wrapper.__name__ = function_name
        return wrapper
    
    def _coerce_ctypes_arg(self, arg_typ, arg):
        # If the argument is already the exact expected ctypes type, keep it.
        try:
            if isinstance(arg, arg_typ):
                return arg
        except TypeError:
            # arg_typ might not be a proper class; ignore and continue
            pass

        # c_char_p expects bytes; accept str/bytes/c_char_p
        if arg_typ is ctypes.c_char_p:
            if arg is None:
                return None
            if isinstance(arg, ctypes.c_char_p):
                return arg
            if isinstance(arg, (bytes, bytearray)):
                return ctypes.c_char_p(bytes(arg))
            if isinstance(arg, str):
                return ctypes.c_char_p(arg.encode("utf-8"))
            if isinstance(arg, ctypes.c_void_p) and arg.value is not None:
                return ctypes.c_char_p(arg.value)
            # fall through to generic conversion error

        # Pointer types: if we got some pointer/byref already, pass it through;
        # if it's a void* or address-int, cast it.
        try:
            is_ptr_type = issubclass(arg_typ, ctypes._Pointer)
        except TypeError:
            is_ptr_type = False

        if is_ptr_type:
            if isinstance(arg, ctypes._Pointer):
                return ctypes.cast(arg, arg_typ)
            if isinstance(arg, ctypes.c_void_p):
                return ctypes.cast(arg, arg_typ)
            # Many ctypes "byref" objects aren't instances of _Pointer; let them through.
            return arg

        # Simple scalars: unwrap .value if it's a ctypes scalar; also accept numpy scalars
        if isinstance(arg, ctypes._SimpleCData):
            val = arg.value
        else:
            val = arg
        # numpy scalar -> Python scalar
        if hasattr(val, "item"):
            try:
                val = val.item()
            except Exception:
                pass
        # Booleans: normalize
        if arg_typ is ctypes.c_bool:
            val = bool(val)

        return arg_typ(val)
    
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
        return int(self._function_pointers[f"Get{enum_value_name}"]())

    # Code generation methods
    def _get_function_def(self, header_code, function_name):
        import re, ctypes

        # 1) Find start after "ORT_API2_STATUS(function_name,"
        pat = re.compile(rf"ORT_API2_STATUS\(\s*{re.escape(function_name)}\s*,", re.DOTALL)
        m = pat.search(header_code)
        if not m:
            raise RuntimeError(f"Couldn't parse ORT header file (couldn't find function {function_name} in header)")

        i = m.end()
        depth = 1  # we are inside the outer '('
        args_chars = []
        while i < len(header_code):
            ch = header_code[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    break
            args_chars.append(ch)
            i += 1
        if depth != 0:
            raise RuntimeError(f"Unbalanced parentheses for {function_name}")

        args_str = "".join(args_chars)

        # 2) Strip SAL annotations completely (keep lowercase like size_t intact)
        args_str = re.sub(
            r"(?<![A-Za-z0-9_])_[A-Z][A-Za-z0-9_]*\s*\([^()]*\)",
            " ",
            args_str,
        )
        # if SAL stripping in older builds already produced ORTCHAR, put the _T back
        args_str = re.sub(
            r"(?<![A-Za-z0-9_])_[A-Z][A-Za-z0-9_]*_?",
            " ",
            args_str,
        )
        args_str = re.sub(r"\bORTCHAR\b(?=\s*\*)", "ORTCHAR_T", args_str)
        # 3) Normalize whitespace
        args_str = re.sub(r"\s+", " ", args_str.strip())

        # 4) Parse arguments
        arg_names = []
        ORTCAPIInterface._function_signatures.setdefault(function_name, [])

        for raw in [a.strip() for a in args_str.split(",") if a.strip() and a.strip() != "void"]:
            # var name is the last identifier
            mname = re.search(r"([A-Za-z_][A-Za-z0-9_]*)$", raw)
            if not mname:
                raise RuntimeError(f"Couldn't parse ORT header file (couldn't parse argument '{raw}' of function {function_name})")
            var_name = mname.group(1)
            type_part = raw[:mname.start()].strip()

            # Remove C qualifiers that can precede the base type
            # (IMPORTANT: strip 'enum' so base becomes 'OrtAllocatorType', 'OrtMemType', etc.)
            type_part = re.sub(r"\b(const|volatile|enum|struct|union)\b", "", type_part).strip()

            # base type (last bare identifier) and any pointer stars after it
            mtype = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(.*)$", type_part)
            if not mtype:
                raise RuntimeError(f"Couldn't parse ORT header file (couldn't parse argument '{raw}' of function {function_name})")
            type_name, ptrs_tail = mtype.group(1), mtype.group(2)
            num_indirection = ptrs_tail.count("*")

            arg_names.append(var_name)

            # Map to ctypes converters
            ctypes_mapping = {
                "int64_t": ctypes.c_int64,
                "size_t": ctypes.c_size_t,
            }

            if type_name in ctypes_mapping and num_indirection == 0:
                ctypes_type = ctypes_mapping[type_name]

            elif type_name == "char" and num_indirection >= 1:
                # char*, char** …
                if num_indirection >= 2:
                    # char**: pass through; caller should supply (c_char_p * n)(...)
                    ctypes_type = (lambda x: x)
                else:
                    # char*: accept None | c_char_p | bytes/bytearray | str | void*
                    def _to_c_char_p(x):
                        if x is None:
                            return None
                        if isinstance(x, ctypes.c_char_p):
                            return x
                        if isinstance(x, (bytes, bytearray)):
                            return ctypes.c_char_p(bytes(x))
                        if isinstance(x, str):
                            return ctypes.c_char_p(x.encode("utf-8"))
                        if isinstance(x, ctypes.c_void_p) and x.value is not None:
                            return ctypes.c_char_p(x.value)
                        # last-resort: try bytes()
                        try:
                            return ctypes.c_char_p(bytes(x))
                        except Exception as e:
                            raise TypeError(f"char* argument must be bytes or str, got {type(x).__name__}") from e
                    ctypes_type = _to_c_char_p

            elif hasattr(ctypes, f"c_{type_name}") and num_indirection == 0:
                ctypes_type = getattr(ctypes, f"c_{type_name}")

            elif type_name in ORTCAPIInterface.enums_to_expose:
                def _enum_coerce(x):
                    # if caller passed the name, look it up
                    if isinstance(x, str):
                        return self.get_enum_value(x)  # already a Python int
                    # if caller passed a ctypes scalar, unwrap
                    if isinstance(x, ctypes._SimpleCData):
                        return int(x.value)
                    # otherwise force to int (handles Python ints, numpy ints)
                    try:
                        return int(x)
                    except Exception:
                        raise TypeError(f"Enum argument must be a name or integer, got {type(x).__name__}")
                ctypes_type = _enum_coerce

            elif num_indirection > 0:
                # Any pointer type (including const void*): pass through; caller supplies byref/pointer/buffer
                ctypes_type = (lambda x: x)
            elif type_name == "ORTCHAR_T" and num_indirection >= 1:
                if os.name == "nt":
                    # Windows uses wide chars
                    ctypes_type = (lambda x: x if isinstance(x, ctypes.c_wchar_p)
                                else ctypes.c_wchar_p(x))
                else:
                    # POSIX uses narrow chars
                    ctypes_type = (lambda x: x if isinstance(x, ctypes.c_char_p)
                                else ctypes.c_char_p(x))
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
        # keep your ctypes signature registration
        ORTCAPIInterface._function_signatures[f"Release{object_name}"] = [lambda x: x]

        # ---- Special case: CUDAProviderOptions (v2 vs v1) ----
        if object_name == "CUDAProviderOptions":
            return r'''
    #if defined(ORT_API_VERSION) && ORT_API_VERSION >= 11
    extern "C" void ReleaseCUDAProviderOptions(OrtCUDAProviderOptionsV2* input) {
        if (ort_api && ort_api->ReleaseCUDAProviderOptions) {
            ort_api->ReleaseCUDAProviderOptions(input);
        }
    }
    #else
    // ORT CUDA EP v1: options is a plain struct; nothing to free.
    // Keep a compatible symbol that's a no-op so callers can always "release".
    extern "C" void ReleaseCUDAProviderOptions(OrtCUDAProviderOptions* /*input*/) {
        // no-op on older ORT
    }
    #endif
    '''
        # ---- Back-compat shim: KernelSession is gone; map to Session ----
        if object_name == "KernelSession":
            return r'''
    extern "C" void ReleaseKernelSession(OrtSession* input) {
        ort_api->ReleaseSession(input);
    }
    '''

        # ---- Generic case ----
        return f'''
    extern "C" void Release{object_name}(Ort{object_name}* input) {{
        ort_api->Release{object_name}(input);
    }}
    '''

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
