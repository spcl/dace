# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes.util
import glob
import importlib.util
import os

import dace.library


def _wheel_root() -> str | None:
    """Directory of the ``nvidia-cudnn-cu*`` wheel, which is how pip installations of torch ship cuDNN."""
    try:
        spec = importlib.util.find_spec("nvidia.cudnn")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


def _wheel_library() -> str | None:
    root = _wheel_root()
    if root is None:
        return None
    candidates = sorted(glob.glob(os.path.join(root, "lib", "libcudnn.so*")), key=len)
    return candidates[0] if candidates else None


def _includes() -> list[str]:
    root = _wheel_root()
    include = os.path.join(root, "include") if root else None
    return [include] if include and os.path.isfile(os.path.join(include, "cudnn.h")) else []


def _libraries() -> list[str]:
    # The wheel has no unversioned ``libcudnn.so`` link, so the library is given by its full path
    wheel = _wheel_library()
    return [wheel] if wheel else ["cudnn"]


@dace.library.environment
class cuDNN:
    """
    Environment for the NVIDIA cuDNN deep neural network primitive library. A system installation is used through
    ``-lcudnn``; otherwise the headers and library of the ``nvidia-cudnn-cu*`` wheel (installed alongside torch by pip)
    are used.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = _includes
    cmake_libraries = _libraries
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {"frame": ["../include/dace_cudnn.h"], "cuda": ["../include/dace_cudnn.h"]}
    state_fields = ["dace::ml::CudnnHandle cudnn_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def handle_setup_code(node):
        return (
            dace.library.reject_gpu_location(node)
            + """\
cudnnHandle_t &__dace_cudnn_handle = __state->cudnn_handle.Get();
dace::ml::CheckCudnnError(cudnnSetStream(__dace_cudnn_handle, __dace_current_stream));\n"""
        )

    @staticmethod
    def _find_library():
        path = ctypes.util.find_library("cudnn") or _wheel_library()
        if path:
            return [path]
        return []

    @staticmethod
    def is_installed():
        return len(cuDNN._find_library()) > 0
