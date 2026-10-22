# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""mimalloc as the allocator of CPU heap arrays (``compiler.cpu.heap_allocator``)."""

import ctypes.util
import os

from dace import config, library


@library.environment
class MiMalloc:
    """Links mimalloc and routes ``dace::aligned_new_array`` / ``aligned_delete_array`` to ``mi_malloc_aligned`` /
    ``mi_free`` (``DACE_MIMALLOC``). Found through ``MIMALLOC_ROOT`` (``$MIMALLOC_ROOT/lib``) or the loader path."""

    cmake_minimum_version = None
    cmake_packages: list[str] = []
    cmake_variables: dict[str, str] = {}
    cmake_includes: list[str] = []
    cmake_compile_flags = ["-DDACE_MIMALLOC"]
    cmake_link_flags: list[str] = []
    cmake_files: list[str] = []

    headers: list[str] = []
    state_fields: list[str] = []
    init_code = ""
    finalize_code = ""
    dependencies: list[str] = []

    @staticmethod
    def library_path() -> str | None:
        """The mimalloc shared library to link, or None when it is not installed."""
        if "MIMALLOC_ROOT" in os.environ:
            prefix = str(config.Config.get("compiler", "library_prefix"))
            suffix = str(config.Config.get("compiler", "library_extension"))
            for libdir in ("lib", "lib64"):
                path = os.path.join(os.environ["MIMALLOC_ROOT"], libdir, f"{prefix}mimalloc.{suffix}")
                if os.path.isfile(path):
                    return path
        return ctypes.util.find_library("mimalloc")

    @staticmethod
    def cmake_libraries():
        path = MiMalloc.library_path()
        return [path] if path is not None and os.path.isabs(path) else ["mimalloc"]

    @staticmethod
    def enabled() -> bool:
        """Whether CPU heap arrays are allocated with mimalloc, per ``compiler.cpu.heap_allocator``."""
        choice = str(config.Config.get("compiler", "cpu", "heap_allocator"))
        if choice == "mimalloc":
            return True
        return choice == "auto" and MiMalloc.library_path() is not None
