# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Which implementation a tile library node lowers to.

The vectorizer stamps ``node.target_isa`` and sets ``node.implementation = select_tile_implementation(node, state)``
before the nodes are expanded. The choice depends only on the target ISA, the tile rank and the node's operands, all
known at that point, so it is a function rather than an ``Auto`` expansion that re-decides at expansion time.
"""
import enum
import functools
import platform

import dace
from dace.sdfg import nodes


class ISA(enum.Enum):
    """Target instruction set a K=1 tile lowers to."""
    AUTO = enum.auto()  #: resolve to the host's best ISA at expansion
    AVX512 = enum.auto()
    AVX2 = enum.auto()
    ARM_SVE = enum.auto()
    ARM_NEON = enum.auto()
    SCALAR = enum.auto()  #: portable scalar reference
    CUDA = enum.auto()  #: GPU half2 (implies device=GPU)


#: Implementation each target ISA lowers a K=1 tile to; K>=2 is always ``pure``.
ISA_TO_IMPL = {
    ISA.AVX512: "avx512",
    ISA.AVX2: "avx2",
    ISA.ARM_SVE: "sve",
    ISA.ARM_NEON: "neon",
    ISA.CUDA: "cuda",
    ISA.SCALAR: "scalar",
}

#: The host SIMD ISAs. Pinning one the host cannot execute compiles (the backend adds its own ``-m`` flag) and then
#: faults at runtime, so :func:`select_tile_implementation` refuses it. ``SCALAR`` always runs; ``CUDA`` is a device ISA
#: the schedule gates.
CPU_SIMD_ISAS = frozenset({ISA.AVX512, ISA.AVX2, ISA.ARM_SVE, ISA.ARM_NEON})

#: The host ISAs in the order ``AUTO`` prefers them.
ISA_PREFERENCE = (ISA.AVX512, ISA.AVX2, ISA.ARM_SVE, ISA.ARM_NEON)


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
def host_supported_isas() -> frozenset[ISA]:
    """The target ISAs the host can execute; an ISA implies the weaker ones of its architecture."""
    machine = platform.machine().lower()
    flags = cpu_flags()
    if machine in ("x86_64", "amd64", "i386", "i686"):
        if "avx512f" in flags:
            return frozenset({ISA.AVX512, ISA.AVX2, ISA.SCALAR})
        if "avx2" in flags:
            return frozenset({ISA.AVX2, ISA.SCALAR})
    elif machine in ("aarch64", "arm64"):
        if "sve" in flags:
            return frozenset({ISA.ARM_SVE, ISA.ARM_NEON, ISA.SCALAR})
        return frozenset({ISA.ARM_NEON, ISA.SCALAR})
    return frozenset({ISA.SCALAR})


def detect_host_isa() -> ISA:
    """The best target ISA the host executes, which is what ``ISA.AUTO`` resolves to."""
    supported = host_supported_isas()
    return next((isa for isa in ISA_PREFERENCE if isa in supported), ISA.SCALAR)


def has_complex_operand(node: nodes.LibraryNode, parent_state: dace.SDFGState) -> bool:
    """Whether any connected operand or output of the tile op is complex."""
    sdfg = parent_state.sdfg
    for edge in (*parent_state.in_edges(node), *parent_state.out_edges(node)):
        if edge.data is None or edge.data.data is None:
            continue
        desc = sdfg.arrays.get(edge.data.data)
        if desc is not None and desc.dtype in (dace.dtypes.complex64, dace.dtypes.complex128):
            return True
    return False


def select_tile_implementation(node: nodes.LibraryNode, parent_state: dace.SDFGState) -> str:
    """The implementation of ``node`` for the target ISA it carries.

    A tile of more than one dim lowers ``pure``, and so does every node the backend headers cannot express
    (:meth:`~dace.libraries.tileops.nodes.tile_op.TileOp.can_lower_to_isa`) and a node with a complex operand on a CPU
    ISA: a packed multiply has no SIMD form, so the scalar loop over ``std::complex`` is the correct lowering
    everywhere. CUDA carries complex natively and keeps its path. The rest take the implementation of their ISA.

    :param node: The tile library node (carries ``widths`` and ``target_isa``).
    :param parent_state: The state holding ``node``.
    :returns: A name in ``node.implementations``.
    :raises ValueError: If the target ISA is a host ISA the host cannot execute.
    """
    target_isa = node.target_isa
    if target_isa is ISA.AUTO:
        target_isa = detect_host_isa()
    implementation = ISA_TO_IMPL.get(target_isa)
    if len(node.widths) != 1 or implementation not in node.implementations:
        return "pure"
    if (target_isa is not ISA.CUDA
            and has_complex_operand(node, parent_state)) or not node.can_lower_to_isa(parent_state, parent_state.sdfg):
        return "pure"
    if target_isa in CPU_SIMD_ISAS and target_isa not in host_supported_isas():
        raise ValueError(f"tile-op target_isa={target_isa.name} is not executable on this host "
                         f"(supported: {sorted(isa.name for isa in host_supported_isas())}). Vectorization enforces "
                         f"arch-native: use ISA.AUTO to target the host, or pick a supported ISA.")
    return implementation
