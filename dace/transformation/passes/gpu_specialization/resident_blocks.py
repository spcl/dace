# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""How many thread blocks of a given width the target GPU holds resident at once.

A grid of that many blocks fills the device once: a grid-strided kernel launches no more, and a buffer with one
slot per block is sized by it. The device is the one visible when the SDFG is compiled, asked through the GPU
runtime's attribute query (microseconds, so a pass asks once per run and hands the limits down); without one, the
``compiler.cuda.device_*`` configuration describes it.
"""

import ctypes
import warnings
from dataclasses import dataclass

from dace.config import Config

#: The multiprocessor count assumed when no GPU is visible and ``compiler.cuda.device_multiprocessors`` is unset.
FALLBACK_MULTIPROCESSORS = 128

#: ``(multiprocessors, threads per multiprocessor, blocks per multiprocessor)`` attribute ids of each runtime: the
#: ``cudaDeviceAttr`` values, and the ``hipDeviceAttribute_t`` ones (fixed since ROCm 5).
ATTRIBUTES = {"cuda": (16, 39, 106), "hip": (63, 57, 25)}


@dataclass(frozen=True, slots=True)
class DeviceLimits:
    """The residency limits of one GPU; a block limit of 0 is none."""

    multiprocessors: int
    threads_per_multiprocessor: int
    blocks_per_multiprocessor: int

    def resident_blocks(self, block_threads: int) -> int:
        """Blocks of ``block_threads`` threads resident at once over all multiprocessors, at least one each."""
        per_multiprocessor = self.threads_per_multiprocessor // block_threads
        if self.blocks_per_multiprocessor > 0:
            per_multiprocessor = min(per_multiprocessor, self.blocks_per_multiprocessor)
        return self.multiprocessors * max(1, per_multiprocessor)


def query_device() -> DeviceLimits | None:
    """The limits of the first visible GPU, from its runtime library; ``None`` without one."""
    from dace.codegen.common import get_gpu_backend, get_gpu_runtime

    try:
        backend = get_gpu_backend()
        runtime = get_gpu_runtime()
    except (RuntimeError, OSError):
        return None
    get = getattr(runtime.library, f"{backend}DeviceGetAttribute")
    values = []
    for attribute in ATTRIBUTES[backend]:
        value = ctypes.c_int(0)
        if get(ctypes.byref(value), ctypes.c_int(attribute), ctypes.c_int(0)) != 0:
            return None
        values.append(value.value)
    multiprocessors, threads, blocks = values
    # HIP derives its block limit as resident threads over the WIDEST block (2048 / 1024 = 2 on CDNA), which is no
    # limit on narrower blocks: on HIP the threads bound residency alone.
    if backend == "hip":
        blocks = 0
    return DeviceLimits(multiprocessors, threads, blocks) if multiprocessors > 0 and threads > 0 else None


def device_limits() -> DeviceLimits:
    """The target GPU's limits: the visible GPU's, unless ``compiler.cuda.device_multiprocessors`` is set or no GPU is
    visible; then the ``compiler.cuda.device_*`` keys, with ``FALLBACK_MULTIPROCESSORS`` for an unset count."""
    multiprocessors = Config.get("compiler", "cuda", "device_multiprocessors")
    threads = Config.get("compiler", "cuda", "device_threads_per_multiprocessor")
    blocks = Config.get("compiler", "cuda", "device_blocks_per_multiprocessor")
    if multiprocessors > 0:
        return DeviceLimits(multiprocessors, threads, blocks)
    visible = query_device()
    if visible is not None:
        return visible
    warnings.warn(
        f"No GPU visible and compiler.cuda.device_multiprocessors unset: assuming {FALLBACK_MULTIPROCESSORS} "
        "multiprocessors for grid and per-block buffer sizes"
    )
    return DeviceLimits(FALLBACK_MULTIPROCESSORS, threads, blocks)


def resident_blocks(block_threads: int) -> int:
    """Blocks of ``block_threads`` threads the target GPU holds resident at once (one device query)."""
    return device_limits().resident_blocks(block_threads)
