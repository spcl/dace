# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes.util
import glob
import importlib.util
import os

import dace.library
from dace.codegen import common
from dace.libraries.comm.environments.mpi import MPI


@dace.library.environment
class NCCL:
    """NCCL, or RCCL when DaCe targets the HIP backend. Both expose the same ``nccl*`` API, so the
    NCCL expansions serve both.

    Communicators are bootstrapped over MPI: every process creates the one of ``MPI_COMM_WORLD``
    when the program is initialized. Each process must drive its own GPU (e.g., through
    ``CUDA_VISIBLE_DEVICES``), as NCCL rejects two ranks on one device. An NCCL installation outside
    the default search paths is found through ``$NCCL_HOME`` (or ``$NCCL_ROOT``) or the
    ``nvidia-nccl-cu*`` Python wheel; ROCm through ``$ROCM_PATH``.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_compile_flags = []
    cmake_files = []

    # The GPU code declares the state struct too, but never uses the context it points to
    headers = {"frame": ["../include/dace_ccl.h"], "cuda": ["../include/dace_ccl_fwd.h"]}
    state_fields = ["dace::comm::NcclContext* nccl;"]
    init_code = """
        __state->nccl = new dace::comm::NcclContext();
        __state->nccl->Create(MPI_COMM_WORLD);
    """
    finalize_code = "delete __state->nccl;"
    dependencies = [MPI]

    @staticmethod
    def uses_rccl() -> bool:
        return common.get_gpu_backend() == "hip"

    @staticmethod
    def nccl_root() -> str | None:
        """The installation prefix of NCCL, or None if it is on the default search paths."""
        for var in ("NCCL_HOME", "NCCL_ROOT"):
            if os.environ.get(var):
                return os.environ[var]
        spec = importlib.util.find_spec("nvidia")
        locations = spec.submodule_search_locations if spec is not None else None
        for location in locations or []:
            if os.path.isfile(os.path.join(location, "nccl", "include", "nccl.h")):
                return os.path.join(location, "nccl")
        return None

    @staticmethod
    def cmake_includes() -> list[str]:
        if NCCL.uses_rccl():
            rocm = os.environ.get("ROCM_PATH")
            return [os.path.join(rocm, "include")] if rocm else []
        root = NCCL.nccl_root()
        return [os.path.join(root, "include")] if root else []

    @staticmethod
    def cmake_libraries() -> list[str]:
        if NCCL.uses_rccl():
            return ["rccl"]
        root = NCCL.nccl_root()
        if root:
            # The wheel ships only the versioned ``libnccl.so.2``, which ``-lnccl`` does not find
            libraries = sorted(glob.glob(os.path.join(root, "lib", "libnccl.so*")))
            if libraries:
                return [libraries[0]]
        return ["nccl"]

    @staticmethod
    def cmake_link_flags() -> list[str]:
        rocm = os.environ.get("ROCM_PATH")
        if NCCL.uses_rccl() and rocm:
            return [f"-L{os.path.join(rocm, 'lib')}"]
        return []

    @staticmethod
    def is_installed() -> bool:
        if NCCL.uses_rccl():
            return ctypes.util.find_library("rccl") is not None
        root = NCCL.nccl_root()
        if root:
            return len(glob.glob(os.path.join(root, "lib", "libnccl.so*"))) > 0
        return ctypes.util.find_library("nccl") is not None
