# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""HIP as a dialect of the CUDA libraries.

The GPU library expansions are written against cuBLAS, cuSOLVER and cuTENSOR. hipBLAS, hipSOLVER and hipTensor mirror
those name for name, so a HIP expansion is the CUDA expansion with its code renamed through the tables below, applied
in order.
"""

import functools
import re

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


def to_hip(code: str, names) -> str:
    """``code`` with every CUDA name of ``names`` replaced by its HIP counterpart."""
    for cuda, hip in names:
        code = code.replace(cuda, hip)
    return code


def rename_code(result, names) -> None:
    """Renames the C++ code of ``result``, a tasklet or an SDFG, recursively."""
    if isinstance(result, nodes.Tasklet):
        result.code.as_string = to_hip(result.code.as_string, names)
        return
    if isinstance(result, nodes.NestedSDFG):
        result = result.sdfg
    for node, _ in result.all_nodes_recursive():
        if isinstance(node, nodes.Tasklet):
            node.code.as_string = to_hip(node.code.as_string, names)


def hip_expansion(cuda_expansion: type[ExpandTransformation], environment: type, names) -> type[ExpandTransformation]:
    """The HIP expansion of ``cuda_expansion``: same graph, code renamed through ``names``, linked to ``environment``."""

    @functools.wraps(cuda_expansion.expansion)
    def expansion(*args, **kwargs):
        result = cuda_expansion.expansion(*args, **kwargs)
        rename_code(result, names)
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
