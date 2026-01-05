# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import os

try:
    import torch.utils.cpp_extension
except ImportError as e:
    raise ImportError("PyTorch is required for torch integration. Install with: pip install dace[ml]") from e

import dace.library

from dace.codegen.common import platform_library_name, get_gpu_backend


@dace.library.environment
class PyTorch:
    """Environment used to build PyTorch C++ Operators."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = torch.utils.cpp_extension.include_paths()

    @staticmethod
    def cmake_libraries():
        """Get the required PyTorch library paths for linking.

        :return: List of library paths for PyTorch CPU libraries.
        :raises RuntimeError: If a required library cannot be found.
        """
        library_names = ["c10", "torch", "torch_cpu", "torch_python"]
        library_paths = []

        for name in library_names:
            for path in torch.utils.cpp_extension.library_paths():
                path = os.path.join(path, platform_library_name(name))
                if os.path.isfile(path):
                    library_paths.append(path)
                    break
            else:
                raise RuntimeError(f"Couldn't locate shared library {name} in PyTorch library paths")

        return library_paths

    cmake_compile_flags = ["-D_GLIBCXX_USE_CXX11_ABI=0"]
    cmake_link_flags = []
    cmake_files = []
    state_fields = []
    dependencies = []

    headers = []
    init_code = ""
    finalize_code = ""


@dace.library.environment
class PyTorchGPU:
    """Environment used to build PyTorch C++ Operators (with CUDA/HIP)."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = torch.utils.cpp_extension.include_paths()

    @staticmethod
    def cmake_libraries():
        """
        Get the required PyTorch library paths for linking with GPU support.

        :return: List of library paths for PyTorch GPU libraries.
        :raises RuntimeError: If a required library cannot be found.
        """
        backend = get_gpu_backend()
        if backend == 'hip':
            library_names = ["c10", "torch", "torch_cpu", "torch_hip", "torch_python", "c10_hip"]
            runtime_lib = "amdhip64"
        else:
            library_names = ["c10", "torch", "torch_cpu", "torch_cuda", "torch_python", "c10_cuda"]
            runtime_lib = "cudart"

        library_paths = []
        for name in library_names:
            for path in torch.utils.cpp_extension.library_paths(device_type=backend):
                path = os.path.join(path, platform_library_name(name))
                if os.path.isfile(path):
                    library_paths.append(path)
                    break
            else:
                raise RuntimeError(f"Couldn't locate shared library {name} in PyTorch library paths")

        return library_paths + [runtime_lib]

    cmake_compile_flags = ["-D_GLIBCXX_USE_CXX11_ABI=0"]
    cmake_link_flags = []
    cmake_files = []
    state_fields = []
    dependencies = []

    headers = []
    init_code = ""
    finalize_code = ""
