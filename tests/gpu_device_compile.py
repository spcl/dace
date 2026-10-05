# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Compile a device translation unit with the toolchain of the selected GPU backend (nvcc or hipcc).

Tests that check what a device compiler accepts, or what it emits, go through here so they hold on CUDA
and HIP alike; the backend comes from ``dace.codegen.common.get_gpu_backend``.
"""
import subprocess
from pathlib import Path
from typing import NamedTuple

import dace
from dace.codegen.common import get_gpu_backend

INCLUDE_DIR = Path(dace.__file__).parent / "runtime" / "include"

# The runtime header a device source needs before ``dace/dace.h``, whichever compiler reads it.
RUNTIME_INCLUDE = "#if defined(__HIPCC__)\n#include <hip/hip_runtime.h>\n#else\n#include <cuda_runtime.h>\n#endif\n"

# How each compiler reports a call no overload resolves uniquely.
AMBIGUOUS_CALL = {"cuda": "more than one instance of overloaded function", "hip": "is ambiguous"}

# The packed half-precision FMA instruction in each backend's device assembly.
PACKED_HALF_FMA = {"cuda": "fma.rn.f16x2", "hip": "v_pk_fma_f16"}


class DeviceBuild(NamedTuple):
    """The compiler run and the file it was asked to write."""
    result: subprocess.CompletedProcess
    output: Path


def first_arch(key: str) -> str:
    """The first architecture configured under ``compiler.cuda.<key>``, or ``native`` (the device of this host),
    which is what DaCe's own build targets when nothing is configured."""
    configured = [arch.strip() for arch in dace.Config.get("compiler", "cuda", key).split(",") if arch.strip()]
    return configured[0] if configured else "native"


def device_compile(source: str, out_dir: Path, assembly: bool = False) -> DeviceBuild:
    """Compile ``source`` (prefixed with :data:`RUNTIME_INCLUDE`) to an object file, or to device assembly
    (PTX for CUDA, GCN/CDNA assembly for HIP) when ``assembly`` is set."""
    src = out_dir / "probe.cu"
    src.write_text(RUNTIME_INCLUDE + source)
    output = out_dir / ("probe.s" if assembly else "probe.o")
    if get_gpu_backend() == "cuda":
        arch = first_arch("cuda_arch")
        cmd = ["nvcc", "-I", str(INCLUDE_DIR), "-std=c++20", "--expt-relaxed-constexpr"]
        cmd += [f"-arch={arch if arch == 'native' else 'sm_' + arch.removeprefix('sm_')}", "-x", "cu"]
        cmd.append("-ptx" if assembly else "-c")
    else:
        cmd = ["hipcc", "-I", str(INCLUDE_DIR), "-std=c++20", f"--offload-arch={first_arch('hip_arch')}", "-x", "hip"]
        cmd += ["-S", "--cuda-device-only"] if assembly else ["-c"]
    result = subprocess.run(cmd + [str(src), "-o", str(output)], capture_output=True, text=True)
    return DeviceBuild(result, output)
