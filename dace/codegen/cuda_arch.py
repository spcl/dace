# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CUDA architectures a CMake build targets when ``compiler.cuda.cuda_arch`` names none and no GPU is visible."""

import functools
import os
import re
import shutil
import subprocess

from dace.config import Config


@functools.lru_cache(maxsize=None, typed=True)
def find_nvcc(root: str | None, path_env: str) -> str | None:
    """``<root>/bin/nvcc`` when it exists, else the ``nvcc`` on ``path_env``; cached per (root, PATH)."""
    if root:
        candidate = os.path.join(root, "bin", "nvcc")
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("nvcc", path=path_env)


@functools.lru_cache(maxsize=None, typed=True)
def nvcc_supported_arches(nvcc: str) -> frozenset | None:
    """The ``sm_XX`` numbers ``nvcc`` can target, from ``--list-gpu-arch``; ``None`` if the probe fails. A newer
    toolkit drops old architectures (CUDA 13 no longer builds ``sm_60``)."""
    try:
        out = subprocess.run([nvcc, "--list-gpu-arch"], capture_output=True, text=True)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return frozenset(int(m) for m in re.findall(r"compute_(\d+)", out.stdout))


#: What nvcc says when ``-arch=native`` has no driver to ask. It does NOT fail: it warns, then
#: compiles for a default architecture of its own, which is older than everything DaCe emits fp16
#: for. The exit code alone therefore cannot tell a resolved GPU from a substituted default.
NO_NATIVE_GPU = "Cannot find valid GPU"


@functools.lru_cache(maxsize=None, typed=True)
def can_use_arch_native(nvcc: str) -> bool:
    """Whether ``nvcc -arch=native`` resolves a local GPU rather than substituting its own default. nvcc exits 0
    either way, so the warning is what has to be read."""
    try:
        out = subprocess.run(
            [nvcc, "-arch=native", "--dryrun", "-x", "cu", "-c", os.devnull, "-o", os.devnull],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return out.returncode == 0 and NO_NATIVE_GPU not in out.stderr


#: Oldest architecture the generated device code can be built for. ``__half`` arithmetic, its
#: comparison operators and the ``half2`` intrinsics the tile ops emit all appear in sm_53; below it
#: <cuda_fp16.h> declares none of them. nvcc's own default is older than this.
MINIMUM_CUDA_ARCH = 53

#: What a build targets when ``-arch=native`` has no GPU to resolve. sm_80 (Ampere) carries everything the
#: runtime emits and is old enough that the cubin still loads on the hardware DaCe is run on.
FALLBACK_CUDA_ARCH = 80


def fallback_arch(supported: set | None) -> int:
    """``FALLBACK_CUDA_ARCH`` unless this toolkit cannot build it, then the oldest it can that the runtime's device
    code still works on."""
    if not supported or FALLBACK_CUDA_ARCH in supported:
        return FALLBACK_CUDA_ARCH
    usable = sorted(arch for arch in supported if arch >= MINIMUM_CUDA_ARCH)
    return usable[0] if usable else MINIMUM_CUDA_ARCH


def cuda_architectures() -> str:
    """The architectures a build targets, ``;``-separated as CMake wants them.

    Empty means ``native``: nvcc resolves the local GPU. ``compiler.cuda.cuda_arch`` overrides it. On a host with no
    visible GPU ``native`` does not fail -- nvcc substitutes a default too old for the fp16 the runtime emits -- so an
    architecture is chosen here instead.
    """
    configured = [arch for arch in map(str.strip, Config.get("compiler", "cuda", "cuda_arch").split(",")) if arch]
    if configured:
        return ";".join(configured)
    root = Config.get("compiler", "cuda", "path") or os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = find_nvcc(root, os.environ.get("PATH", ""))
    if nvcc is None or can_use_arch_native(nvcc):
        return ""
    return str(fallback_arch(nvcc_supported_arches(nvcc)))
