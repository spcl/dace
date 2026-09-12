# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Build and drive the memory microbenchmarks in ``membench.c``; reports the MIN over windows, never the mean, since interference only ever adds time."""
import ctypes
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "membench.c")

# no -O3 memcpy-rewrite (glibc-dependent strategy), no -ffast-math (perturbs the triad)
CFLAGS = ["-O3", "-march=native", "-fno-tree-loop-distribute-patterns", "-fopenmp", "-shared", "-fPIC"]


@dataclass(frozen=True)
class Sample:
    """One measurement: the MIN over accepted windows, qualified by the spread across them."""
    value: float  # ns/hop for the chase, hardware bytes/s for the triad
    spread: float  # (max - min) / min over accepted windows
    windows: int  # accepted (fault-free) windows


def build(cc: Optional[str] = None, out_dir: Optional[str] = None) -> ctypes.CDLL:
    """Compile ``membench.c`` to a shared object and load it. Cached by the caller if desired."""
    cc = cc or os.environ.get("CC", "cc")
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "libmembench.so")
    subprocess.run([cc, *CFLAGS, SOURCE, "-o", out], check=True, capture_output=True, text=True)
    lib = ctypes.CDLL(out)

    lib.chase_setup.restype = ctypes.c_void_p
    lib.chase_setup.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint, ctypes.c_uint]
    lib.chase_verify.restype = ctypes.c_int
    lib.chase_verify.argtypes = [ctypes.c_void_p]
    lib.chase_flush.restype = None
    lib.chase_flush.argtypes = [ctypes.c_void_p]
    lib.chase_run_timed.restype = ctypes.c_uint64
    lib.chase_run_timed.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64)
    ]
    lib.chase_teardown.restype = None
    lib.chase_teardown.argtypes = [ctypes.c_void_p]
    lib.triad_run_timed.restype = None
    lib.triad_run_timed.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64)
    ]
    return lib


def transparent_hugepages_enabled() -> bool:
    """Whether THP backs the arena; if not, L picks up a page-walk term unrelated to the memory level measured."""
    try:
        with open("/sys/kernel/mm/transparent_hugepage/enabled") as handle:
            return "[never]" not in handle.read()
    except OSError:
        return False


def anon_hugepage_bytes() -> int:
    """AnonHugePages currently backing this process, from smaps_rollup."""
    try:
        with open("/proc/self/smaps_rollup") as handle:
            for line in handle:
                if line.startswith("AnonHugePages:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def load_average() -> float:
    return os.getloadavg()[0]


def available_bytes() -> int:
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def check_environment(arena_bytes: int, max_load: float = 0.5) -> List[str]:
    """Reasons this machine cannot produce a trustworthy number right now; empty means go."""
    reasons = []
    load = load_average()
    if load > max_load:
        reasons.append(f"load average {load:.2f} > {max_load} (a contended box inflates every sample)")
    available = available_bytes()
    if available < 4 * arena_bytes:
        reasons.append(f"MemAvailable {available / 2**30:.2f} GiB < 4x arena "
                       f"({4 * arena_bytes / 2**30:.2f} GiB): risks reclaim/swap inside a window")
    if not transparent_hugepages_enabled():
        reasons.append("transparent hugepages are 'never': the arena would use 4 KiB pages and L "
                       "would carry a TLB page-walk term")
    return reasons


def fit_core_mlp(latency_by_chains: Dict[int, float]) -> float:
    """Core MLP budget: L(1 chain) / min ns-per-load over the concurrency sweep (Little's Law floor)."""
    if 1 not in latency_by_chains:
        raise ValueError("the sweep must include the 1-chain (MLP=1) point; it defines L")
    if len(latency_by_chains) < 2:
        raise ValueError("need at least two chain counts to see the latency overlap")
    bad = {c: t for c, t in latency_by_chains.items() if c < 1 or t <= 0}
    if bad:
        raise ValueError(f"invalid sweep points (chains must be >= 1, ns/load > 0): {bad}")
    # exact only once the sweep reaches the flat region; refuse if the minimum sits at the largest chain count
    floor_chains = min(latency_by_chains, key=latency_by_chains.get)
    if floor_chains == max(latency_by_chains):
        raise ValueError(f"sweep truncated: the minimum ns/load is at the largest chain count "
                         f"({floor_chains}); extend the sweep past the knee so the flat region is visible")
    return latency_by_chains[1] / min(latency_by_chains.values())
