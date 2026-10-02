# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Which implementation a tile library node lowers to.

The vectorizer stamps ``node.target_isa`` and sets ``node.implementation = select_tile_implementation(node, state)``
before the nodes are expanded. The choice depends only on the target ISA, the tile rank and the node's operands, all
known at that point, so it is a function rather than an ``Auto`` expansion that re-decides at expansion time.
"""
import functools
import platform

import dace
from dace.sdfg import nodes

#: Implementation each target ISA lowers a K=1 tile to; K>=2 is always ``pure``.
ISA_TO_IMPL = {
    "AVX512": "avx512",
    "AVX2": "avx2",
    "ARM_SVE": "sve",
    "ARM_NEON": "neon",
    "CUDA": "cuda",
    "SCALAR": "scalar",
}

#: The host SIMD ISAs. Pinning one the host cannot execute compiles (the backend adds its own ``-m`` flag) and then
#: faults at runtime, so :func:`select_tile_implementation` refuses it. ``SCALAR`` always runs; ``CUDA`` is a device ISA
#: the schedule gates.
CPU_SIMD_ISAS = frozenset({"AVX512", "AVX2", "ARM_SVE", "ARM_NEON"})

#: The host ISAs in the order ``AUTO`` prefers them.
ISA_PREFERENCE = ("AVX512", "AVX2", "ARM_SVE", "ARM_NEON")

#: Ops with no per-ISA op code. They lower to a per-lane ``std::<fn>`` call in the pure loop, which the compiler's
#: vector-math library captures. (``sin``, ``cos``, ``exp``, ``log``, ``sqrt`` and ``tanh`` have op codes.)
PURE_ONLY_MATH_OPS = frozenset(
    {"atan2", "hypot", "fmod", "tan", "asin", "acos", "atan", "sinh", "cosh", "pow", "ipow", "**"})


@functools.lru_cache(maxsize=1, typed=True)
def cpu_flags() -> frozenset[str]:
    """The CPU feature flags ``/proc/cpuinfo`` lists; empty where it cannot be read (e.g. macOS)."""
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith(("flags", "Features")):
                    return frozenset(line.split(":", 1)[1].split())
    except OSError:
        pass
    return frozenset()


@functools.lru_cache(maxsize=1, typed=True)
def host_supported_isas() -> frozenset[str]:
    """The target ISAs the host can execute; an ISA implies the weaker ones of its architecture."""
    machine = platform.machine().lower()
    flags = cpu_flags()
    if machine in ("x86_64", "amd64", "i386", "i686"):
        if "avx512f" in flags:
            return frozenset({"AVX512", "AVX2", "SCALAR"})
        if "avx2" in flags:
            return frozenset({"AVX2", "SCALAR"})
    elif machine in ("aarch64", "arm64"):
        if "sve" in flags:
            return frozenset({"ARM_SVE", "ARM_NEON", "SCALAR"})
        return frozenset({"ARM_NEON", "SCALAR"})
    return frozenset({"SCALAR"})


def detect_host_isa() -> str:
    """The best target ISA the host executes, which is what ``target_isa="AUTO"`` resolves to."""
    supported = host_supported_isas()
    return next((isa for isa in ISA_PREFERENCE if isa in supported), "SCALAR")


def has_complex_operand(node: nodes.LibraryNode, parent_state: dace.SDFGState | None) -> bool:
    """Whether any connected operand or output of the tile op is complex; an unknown context answers ``False``."""
    if parent_state is None:
        return False
    sdfg = parent_state.sdfg
    for edge in (*parent_state.in_edges(node), *parent_state.out_edges(node)):
        if edge.data is None or edge.data.data is None:
            continue
        desc = sdfg.arrays.get(edge.data.data)
        if desc is not None and desc.dtype in (dace.dtypes.complex64, dace.dtypes.complex128):
            return True
    return False


def select_tile_implementation(node: nodes.LibraryNode, parent_state: dace.SDFGState | None = None) -> str:
    """The implementation of ``node`` for the target ISA it carries.

    K>=2 and the ops without an op code lower to ``pure``. So does a complex operand on a CPU ISA: a packed multiply
    has no SIMD form, so the scalar loop over ``std::complex`` is the correct lowering everywhere (CUDA carries
    complex natively and keeps its path). Anything else takes the implementation of its ISA, falling back to ``pure``
    where the node defines none.

    :param node: The tile library node (carries ``widths`` and ``target_isa``).
    :param parent_state: The state holding ``node``, to read the operand dtypes from.
    :returns: A name in ``node.implementations``.
    :raises ValueError: If the target ISA is a host ISA the host cannot execute.
    """
    if len(node.widths) != 1:
        return "pure"
    if getattr(node, "op", None) in PURE_ONLY_MATH_OPS:
        return "pure"
    target_isa = getattr(node, "target_isa", "SCALAR")
    if target_isa == "AUTO":
        target_isa = detect_host_isa()
    if target_isa in CPU_SIMD_ISAS and target_isa not in host_supported_isas():
        raise ValueError(f"tile-op target_isa={target_isa!r} is not executable on this host "
                         f"(supported: {sorted(host_supported_isas())}). Vectorization enforces "
                         f"arch-native: use ISA.AUTO to target the host, or pick a supported ISA.")
    if target_isa != "CUDA" and has_complex_operand(node, parent_state):
        return "pure"
    implementation = ISA_TO_IMPL.get(target_isa, "pure")
    return implementation if implementation in node.implementations else "pure"
