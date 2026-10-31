# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""HIP as a dialect of the CUDA libraries.

The GPU library expansions are written against cuBLAS, cuSOLVER and cuTENSOR. hipBLAS, hipSOLVER and hipTensor mirror
those name for name, so a HIP expansion is the CUDA expansion with its code renamed through the tables below, applied
in order.
"""

import ctypes.util
import functools
import os
import pathlib
import re
import shutil

from dace import nodes
from dace.transformation.transformation import ExpandTransformation

#: Runtime and data-type names every library shares.
RUNTIME = (
    ("cuDoubleComplex", "hipDoubleComplex"),
    ("cuComplex", "hipComplex"),
    ("CUDA_R_", "HIP_R_"),
    ("CUDA_C_", "HIP_C_"),
    ("cudaMalloc", "hipMalloc"),
    ("cudaMemcpy", "hipMemcpy"),
    ("cudaMemset", "hipMemset"),
    ("cudaFree", "hipFree"),
)

#: cuBLAS to hipBLAS. hipBLAS has no tensor-op algorithm; its default picks one where the device has it.
BLAS = (
    ("CUBLAS_GEMM_DEFAULT_TENSOR_OP", "HIPBLAS_GEMM_DEFAULT"),
    ("cublas", "hipblas"),
    ("CUBLAS_", "HIPBLAS_"),
) + RUNTIME

#: cuSOLVER to hipSOLVER. cuSOLVER takes cuBLAS's fill-mode and operation enums, hipSOLVER its own.
SOLVER = (
    ("cusolverDn", "hipsolverDn"),
    ("CUSOLVER_", "HIPSOLVER_"),
    ("CUBLAS_", "HIPSOLVER_"),
) + RUNTIME


#: cuTENSOR 2 to hipTensor 2.
TENSOR = (
    ("cutensor", "hiptensor"),
    ("CUTENSOR_", "HIPTENSOR_"),
) + RUNTIME

#: The GPU BLAS each GPU solver implementation pairs with, for the transposes around a solver call.
GPU_SOLVER_BLAS = {"cuSolverDn": "cuBLAS", "hipSolverDn": "hipBLAS"}


def rocm_roots() -> list[pathlib.Path]:
    """The ROCm installation roots: ``ROCM_PATH``, ``HIP_PATH``, then the root of the ``hipcc`` on ``PATH``."""
    roots = [pathlib.Path(os.environ[v]) for v in ("ROCM_PATH", "HIP_PATH") if os.environ.get(v)]
    hipcc = shutil.which("hipcc")
    if hipcc is not None:
        roots.append(pathlib.Path(hipcc).resolve().parent.parent)
    return roots


def find_rocm_library(name: str) -> str | None:
    """The path of ROCm library ``name`` (``hipblas``): the loader's search first, then each ROCm root, whose
    ``lib`` directory the loader rarely lists."""
    found = ctypes.util.find_library(name)
    if found is not None:
        return found
    for root in rocm_roots():
        for library in sorted((root / "lib").glob(f"lib{name}.so*")):
            return str(library)
    return None


def to_hip(code: str, names) -> str:
    """``code`` with every CUDA name of ``names`` replaced by its HIP counterpart."""
    for cuda, hip in names:
        code = code.replace(cuda, hip)
    return code


def rename_code(result, names, environments: dict[str, str] | None = None) -> None:
    """Renames the C++ code of ``result``, a tasklet or an SDFG, recursively, and retargets the environments its
    tasklets name through ``environments`` (CUDA class path to HIP class path)."""
    if isinstance(result, nodes.NestedSDFG):
        result = result.sdfg
    found = [result] if isinstance(result, nodes.Tasklet) else [node for node, _ in result.all_nodes_recursive()]
    for node in found:
        if not isinstance(node, nodes.Tasklet):
            continue
        node.code.as_string = to_hip(node.code.as_string, names)
        if environments and node.environments:
            node.environments = {environments.get(env, env) for env in node.environments}


def hip_expansion(cuda_expansion: type[ExpandTransformation], environment: type, names) -> type[ExpandTransformation]:
    """The HIP expansion of ``cuda_expansion``: same graph, code renamed through ``names``, linked to ``environment``."""

    # The CUDA expansion tags the tasklets it builds with its own environments, which link the CUDA library
    retarget = {env.full_class_path(): environment.full_class_path() for env in cuda_expansion.environments}

    @functools.wraps(cuda_expansion.expansion)
    def expansion(*args, **kwargs):
        result = cuda_expansion.expansion(*args, **kwargs)
        rename_code(result, names, retarget)
        return result

    # ExpandGemmCuBLAS -> ExpandGemmHipBLAS, cuFFTFFTExpansion -> hipFFTFFTExpansion
    name = re.sub(r"([Cc])u(?=[A-Z])", lambda m: "Hip" if m[1] == "C" else "hip", cuda_expansion.__name__)
    attributes = {
        "environments": [environment],
        "expansion": staticmethod(expansion),
        "__module__": cuda_expansion.__module__,
    }
    from dace import library  # Avoid import loop

    return library.expansion(type(name, (ExpandTransformation,), attributes))
