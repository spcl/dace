import os
import dace.library

CONDA_HOME = "/home/orausch/.local/opt/miniconda3/envs/dace"

@dace.library.environment
class ONNXRuntime:
    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = [
        "/home/orausch/sources/onnx", # pick up the onnx source headers
        "/home/orausch/sources/onnx/build-no-ml", # pick up the compiled protobuf headers
        "/home/orausch/sources/dace/dist-cpu/include"
    ]
    cmake_libraries = [
        "/home/orausch/sources/dace/dist-cpu/libonnxruntime.so",
        "/home/orausch/sources/onnx/build-no-ml/libonnx_proto.a",
        "protobuf",
    ]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = [
        "onnx/onnx_pb.h",
        "onnxruntime_c_api.h",
        "cpu_provider_factory.h",
        "cuda_provider_factory.h",
    ]
    init_code = ""
    finalize_code = ""
