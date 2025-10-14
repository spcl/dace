# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import os

try:
    import torch.utils.cpp_extension
except ImportError as e:
    raise ImportError("PyTorch is required for torch integration. Install with: pip install dace[ml]") from e

import dace.library

from dace.util import platform_library_name


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

        Returns:
            List of library paths for PyTorch CPU libraries

        Raises:
            RuntimeError: If a required library cannot be found
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
class PyTorchCUDA:
    """Environment used to build PyTorch C++ Operators (with CUDA)."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = torch.utils.cpp_extension.include_paths()

    @staticmethod
    def cmake_libraries():
        """Get the required PyTorch library paths for linking with CUDA support.

        Returns:
            List of library paths for PyTorch CUDA libraries

        Raises:
            RuntimeError: If a required library cannot be found
        """
        library_names = ["c10", "torch", "torch_cpu", "torch_cuda", "torch_python", "c10_cuda"]
        library_paths = []

        for name in library_names:
            for path in torch.utils.cpp_extension.library_paths(cuda=True):
                path = os.path.join(path, platform_library_name(name))
                if os.path.isfile(path):
                    library_paths.append(path)
                    break
            else:
                raise RuntimeError(f"Couldn't locate shared library {name} in PyTorch library paths")

        return library_paths + ["cudart"]

    cmake_compile_flags = ["-D_GLIBCXX_USE_CXX11_ABI=0"]
    cmake_link_flags = []
    cmake_files = []
    state_fields = []
    dependencies = []

    headers = []
    init_code = ""
    finalize_code = ""
