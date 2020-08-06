import os
import dace.library
from ctypes import CDLL

if 'ORT_ROOT' not in os.environ:
    raise ValueError(
        "This environment expects the environment variable "
        "ORT_ROOT to be set to the root of the patched onnxruntime repository "
        "(https://github.com/orausch/onnxruntime).\n"
        "See the docstring for dace.libraries.onnx.environments.ONNXRuntime for more information."
    )

ORT_PATH = os.environ['ORT_ROOT']
cand_path = os.path.join(ORT_PATH, "build", "Linux", dace.Config.get("compiler", "build_type"))

if os.path.isdir(cand_path):
    ORT_BUILD_PATH = cand_path
else:
    ORT_BUILD_PATH = os.path.join(ORT_PATH, "build", "Linux", "Release")

ORT_DLL_PATH = os.path.join(ORT_BUILD_PATH, "libonnxruntime.so")

@dace.library.environment
class ONNXRuntime:
    """ Environment used to run ONNX operator nodes using ONNX Runtime. This environment expects the environment variable
        ``ORT_ROOT`` to be set to the root of the patched onnxruntime repository (https://github.com/orausch/onnxruntime)

        Furthermore, both the runtime and the protobuf shared libs should be built:

        ``./build.sh --build_shared_lib --parallel --config Release``
        ``mkdir build-protobuf && cd build-protobuf && cmake ../cmake/external/protobuf/cmake -Dprotobuf_BUILD_SHARED_LIBS=ON && make``

        (add ``-jN`` to the make command for parallel builds)
        See ``onnxruntime/BUILD.md`` for more details.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = [
        ORT_BUILD_PATH,
        os.path.join(ORT_PATH, "cmake", "external", "onnx"),
        os.path.join(ORT_PATH, "include"),
        os.path.join(ORT_PATH, "cmake", "external", "protobuf", "src"),
        os.path.join(ORT_PATH, "include", "onnxruntime", "core", "session"),
        os.path.join(ORT_PATH, "include", "onnxruntime", "core", "providers",
                     "cpu"),
        os.path.join(ORT_PATH, "include", "onnxruntime", "core", "providers",
                     "cuda")
    ]
    cmake_libraries = [
        ORT_DLL_PATH
    ]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = [
        "../include/dace_onnx.h",
        "onnxruntime_c_api.h",
        "cpu_provider_factory.h",
        "cuda_provider_factory.h",
    ]
    init_code = ""
    finalize_code = ""
