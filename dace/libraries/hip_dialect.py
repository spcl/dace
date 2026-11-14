# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""HIP as a dialect of the CUDA libraries.

The GPU library expansions are written against the CUDA libraries, which their HIP counterparts mirror name for name,
so a HIP expansion is the CUDA expansion with its code renamed through a table such as ``FFT``, applied in order.
"""

import ctypes.util
import functools
import os
import pathlib
import re
import shutil

from dace import nodes
from dace.transformation.transformation import ExpandTransformation

#: cuFFT to hipFFT: a prefix swap, except the inverse direction.
FFT = (
    ("CUFFT_INVERSE", "HIPFFT_BACKWARD"),
    ("cufft", "hipfft"),
    ("CUFFT_", "HIPFFT_"),
    ("cuFFT", "hipFFT"),
)


def rocm_roots() -> list[pathlib.Path]:
    """The ROCm roots: ``ROCM_PATH``, ``HIP_PATH``, then the root of the ``hipcc`` on ``PATH``.

    :return: The roots, in that order.
    """
    roots = [pathlib.Path(os.environ[v]) for v in ("ROCM_PATH", "HIP_PATH") if os.environ.get(v)]
    hipcc = shutil.which("hipcc")
    if hipcc is not None:
        roots.append(pathlib.Path(hipcc).resolve().parent.parent)
    return roots


def find_rocm_library(name: str) -> str | None:
    """The path of a ROCm library: the loader's search first, then each ROCm root's ``lib``.

    :param name: The library name, e.g. ``hipblas``.
    :return: The path, or ``None``.
    """
    found = ctypes.util.find_library(name)
    if found is not None:
        return found
    for root in rocm_roots():
        for library in sorted((root / "lib").glob(f"lib{name}.so*")):
            return str(library)
    return None


def to_hip(code: str, names) -> str:
    """``code`` with every CUDA name of ``names`` replaced by its HIP counterpart.

    :param code: The CUDA code.
    :param names: ``(CUDA, HIP)`` pairs, exceptions before prefixes.
    :return: The HIP code.
    """
    for cuda, hip in names:
        code = code.replace(cuda, hip)
    return code


def rename_code(result, names) -> None:
    """Renames the C++ code of a tasklet or, recursively, an SDFG to HIP.

    :param result: The tasklet or SDFG.
    :param names: ``(CUDA, HIP)`` pairs.
    """
    if isinstance(result, nodes.Tasklet):
        result.code.as_string = to_hip(result.code.as_string, names)
        return
    if isinstance(result, nodes.NestedSDFG):
        result = result.sdfg
    for node, _ in result.all_nodes_recursive():
        if isinstance(node, nodes.Tasklet):
            node.code.as_string = to_hip(node.code.as_string, names)


def hip_expansion(cuda_expansion: type[ExpandTransformation], environment: type, names) -> type[ExpandTransformation]:
    """The HIP expansion of a CUDA one: the same graph with its code renamed.

    :param cuda_expansion: The CUDA expansion.
    :param environment: The HIP environment to link.
    :param names: ``(CUDA, HIP)`` pairs.
    :return: The HIP expansion.
    """

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
