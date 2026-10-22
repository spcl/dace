# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LIBXSMM: JIT-generated small matrix multiplications."""

import ctypes.util
import os

from dace import config, library


@library.environment
class LIBXSMM:
    """Links LIBXSMM, found through ``LIBXSMM_ROOT`` (``$LIBXSMM_ROOT/{include/libxsmm,lib}``) or the loader path.
    ``libxsmm_?gemm`` falls back to the BLAS ``?gemm_`` symbol for a shape it does not generate, so an expansion
    calling it links OpenBLAS too."""

    cmake_minimum_version = None
    cmake_packages: list[str] = []
    cmake_variables: dict[str, str] = {}
    cmake_compile_flags: list[str] = []
    cmake_link_flags: list[str] = []
    cmake_files: list[str] = []

    headers = ["libxsmm.h"]
    state_fields: list[str] = []
    init_code = "libxsmm_init();"
    finalize_code = ""
    dependencies: list[str] = []

    @staticmethod
    def library_path() -> str | None:
        """The LIBXSMM shared library to link, or None when it is not installed."""
        if "LIBXSMM_ROOT" in os.environ:
            prefix = str(config.Config.get("compiler", "library_prefix"))
            suffix = str(config.Config.get("compiler", "library_extension"))
            for libdir in ("lib", "lib64"):
                path = os.path.join(os.environ["LIBXSMM_ROOT"], libdir, f"{prefix}xsmm.{suffix}")
                if os.path.isfile(path):
                    return path
        return ctypes.util.find_library("xsmm")

    @staticmethod
    def cmake_includes():
        if "LIBXSMM_ROOT" in os.environ:
            return [os.path.join(os.environ["LIBXSMM_ROOT"], "include", "libxsmm")]
        return []

    @staticmethod
    def cmake_libraries():
        path = LIBXSMM.library_path()
        if path is None:
            return ["xsmm"]
        return [path] if os.path.isabs(path) else [f"-l:{path}"]

    @staticmethod
    def is_installed() -> bool:
        return LIBXSMM.library_path() is not None
