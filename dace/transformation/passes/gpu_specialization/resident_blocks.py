# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""How many thread blocks of a given width the target GPU holds resident at once.

A grid of that many blocks fills the device once: a grid-strided kernel launches no more, and a buffer with one
slot per block is sized by it. The device is the one visible when the SDFG is compiled, probed once per process
(``tools/get_device_limits.cpp``); without one, the ``compiler.cuda.device_*`` configuration describes it.
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from dace.config import Config

#: The multiprocessor count assumed when no GPU is visible and ``compiler.cuda.device_multiprocessors`` is unset.
FALLBACK_MULTIPROCESSORS = 128


@dataclass(frozen=True, slots=True)
class DeviceLimits:
    """The residency limits of one GPU."""

    multiprocessors: int
    threads_per_multiprocessor: int
    blocks_per_multiprocessor: int

    def resident_blocks(self, block_threads: int) -> int:
        """Blocks of ``block_threads`` threads resident at once over all multiprocessors, at least one each."""
        per_multiprocessor = self.threads_per_multiprocessor // block_threads
        if self.blocks_per_multiprocessor > 0:  # a runtime that does not report the limit reports 0
            per_multiprocessor = min(per_multiprocessor, self.blocks_per_multiprocessor)
        return self.multiprocessors * max(1, per_multiprocessor)


class DeviceProbe:
    """The visible GPU's limits, probed once (a compile and a run of a small program) and kept."""

    def __init__(self) -> None:
        self.probed = False
        self.found: DeviceLimits | None = None

    def visible(self) -> DeviceLimits | None:
        """The first visible GPU's limits, or ``None`` when there is none or no compiler for it."""
        if not self.probed:
            self.probed = True
            self.found = probe_device()
        return self.found


def probe_device() -> DeviceLimits | None:
    """Compile and run ``get_device_limits.cpp`` with the backend's compiler; ``None`` if any step fails."""
    from dace import codegen
    from dace.codegen.common import get_gpu_backend

    try:
        backend = get_gpu_backend()
    except RuntimeError:
        return None
    home = os.environ.get("ROCM_PATH", "/opt/rocm") if backend == "hip" else os.environ.get("CUDA_HOME", "")
    compiler = shutil.which("hipcc" if backend == "hip" else "nvcc") or os.path.join(
        home, "bin", "hipcc" if backend == "hip" else "nvcc"
    )
    source = os.path.join(os.path.dirname(codegen.__file__), "tools", "get_device_limits.cpp")
    with tempfile.TemporaryDirectory() as folder:
        binary = os.path.join(folder, "device_limits")
        flags = ["-DDACE_PROBE_HIP"] if backend == "hip" else []
        try:
            built = subprocess.run([compiler, *flags, source, "-o", binary], capture_output=True, timeout=300)
            if built.returncode != 0:
                return None
            ran = subprocess.run([binary], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
    fields = ran.stdout.split()
    if ran.returncode != 0 or len(fields) != 3 or not all(field.isdigit() for field in fields):
        return None
    multiprocessors, threads, blocks = (int(field) for field in fields)
    # HIP derives its block limit as resident threads over the WIDEST block (2048 / 1024 = 2 on CDNA), which is no
    # limit on narrower blocks: on HIP the threads bound residency alone.
    if backend == "hip":
        blocks = 0
    return DeviceLimits(multiprocessors, threads, blocks) if multiprocessors > 0 and threads > 0 else None


#: The process's probe: the device does not change while it runs.
PROBE = DeviceProbe()


def device_limits() -> DeviceLimits:
    """The target GPU's limits: the visible GPU's, unless ``compiler.cuda.device_multiprocessors`` is set or no GPU is
    visible; then the ``compiler.cuda.device_*`` keys, with ``FALLBACK_MULTIPROCESSORS`` for an unset count."""
    multiprocessors = Config.get("compiler", "cuda", "device_multiprocessors")
    threads = Config.get("compiler", "cuda", "device_threads_per_multiprocessor")
    blocks = Config.get("compiler", "cuda", "device_blocks_per_multiprocessor")
    visible = PROBE.visible() if multiprocessors <= 0 else None
    if visible is not None:
        return visible
    return DeviceLimits(multiprocessors if multiprocessors > 0 else FALLBACK_MULTIPROCESSORS, threads, blocks)


def resident_blocks(block_threads: int) -> int:
    """Blocks of ``block_threads`` threads the target GPU holds resident at once."""
    return device_limits().resident_blocks(block_threads)
