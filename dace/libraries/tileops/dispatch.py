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
from dace import tile_functions
from dace.sdfg import nodes
from dace.sdfg.scope import is_devicelevel_gpu


class ISA(enum.Enum):
    """Target instruction set a K=1 tile lowers to."""

    AUTO = enum.auto()  #: resolve to the host's best ISA at expansion
    AVX512 = enum.auto()
    AVX2 = enum.auto()
    ARM_SVE = enum.auto()
    ARM_NEON = enum.auto()
    SCALAR = enum.auto()  #: portable scalar reference
    CUDA = enum.auto()  #: GPU half2 (implies device=GPU)


class TileGroup(enum.Enum):
    """The threads that execute a tile node together, which decides how it is lowered.

    On a CPU one core runs the whole tile, so every group lowers as ``THREAD`` does. In a GPU kernel ``THREAD`` is one
    thread's register tile, and ``WARP`` and ``BLOCK`` spread the lanes over the threads of a warp or a thread block.
    """

    THREAD = enum.auto()  #: one thread (one SIMD vector on a CPU); what the vectorizer emits
    WARP = enum.auto()  #: the 32 threads of a warp
    BLOCK = enum.auto()  #: the threads of a thread block (one core on a CPU); what ``dace.tile`` calls create


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


def operand_dtypes(node: nodes.LibraryNode, parent_state: dace.SDFGState) -> list[dace.typeclass]:
    """The element types of the connected operands and outputs of the tile op."""
    sdfg = parent_state.sdfg
    edges = (*parent_state.in_edges(node), *parent_state.out_edges(node))
    descs = [sdfg.arrays.get(edge.data.data) for edge in edges if edge.data is not None and edge.data.data is not None]
    return [desc.dtype for desc in descs if desc is not None]


def has_complex_operand(node: nodes.LibraryNode, parent_state: dace.SDFGState) -> bool:
    """Whether any connected operand or output of the tile op is complex."""
    return any(dtype in (dace.dtypes.complex64, dace.dtypes.complex128) for dtype in operand_dtypes(node, parent_state))


def select_tile_implementation(node: nodes.LibraryNode, parent_state: dace.SDFGState) -> str:
    """The implementation of ``node`` for the target ISA it carries.

    A tile of more than one dim lowers ``pure``, and so does every node the backends cannot express
    (:meth:`~dace.libraries.tileops.nodes.tile_op.TileOp.can_lower_to_isa`), a node with a complex operand on a CPU
    ISA (a packed multiply has no SIMD form, so the scalar loop over ``std::complex`` is the correct lowering
    everywhere; CUDA carries complex natively and keeps its path) and a node over an element type no CPU tile function
    takes (``float16``). The rest take the implementation of their ISA.

    A ``WARP`` or ``BLOCK`` node in a GPU kernel lowers ``block``, spread over the threads of its group.

    :param node: The tile library node (carries ``widths``, ``target_isa`` and ``group``).
    :param parent_state: The state holding ``node``.
    :returns: A name in ``node.implementations``.
    :raises ValueError: If the target ISA is a host ISA the host cannot execute.
    """
    if node.group is not TileGroup.THREAD and is_devicelevel_gpu(parent_state.sdfg, parent_state, node):
        return "block"
    return lane_implementation(node, parent_state)


def lane_implementation(node: nodes.LibraryNode, parent_state: dace.SDFGState) -> str:
    """The implementation one thread runs the lanes of ``node`` with: ``pure`` or the backend of its target ISA. A
    ``block`` lowering runs it on each thread's share of the lanes."""
    target_isa = node.target_isa
    if target_isa is ISA.AUTO:
        target_isa = detect_host_isa()
    implementation = isa_implementation(node, parent_state, target_isa)
    if implementation != "pure" and target_isa in CPU_SIMD_ISAS and target_isa not in host_supported_isas():
        raise ValueError(
            f"tile-op target_isa={target_isa.name} is not executable on this host "
            f"(supported: {sorted(isa.name for isa in host_supported_isas())}). Vectorization enforces "
            f"arch-native: use ISA.AUTO to target the host, or pick a supported ISA."
        )
    return implementation


def isa_implementation(node: nodes.LibraryNode, parent_state: dace.SDFGState, target_isa: ISA) -> str:
    """The backend of ``target_isa`` for ``node``, or ``pure`` where that backend cannot express it; whether the host
    executes it is the caller's question."""
    implementation = ISA_TO_IMPL.get(target_isa)
    if len(node.widths) != 1 or implementation not in node.implementations:
        return "pure"
    if (target_isa is not ISA.CUDA and has_complex_operand(node, parent_state)) or not node.can_lower_to_isa(
        parent_state, parent_state.sdfg
    ):
        return "pure"
    if implementation in tile_functions.ISAS and any(
        dtype.ctype not in tile_functions.ELEMENT_TYPES for dtype in operand_dtypes(node, parent_state)
    ):
        return "pure"
    return implementation


#: The host ISAs with tile functions (:mod:`dace.tile_functions`), widest first.
FUNCTION_ISA_PREFERENCE = (ISA.AVX512, ISA.AVX2, ISA.ARM_NEON)

#: The ISAs whose tile nodes lower to tile functions.
FUNCTION_ISAS = (*FUNCTION_ISA_PREFERENCE, ISA.SCALAR)


def widest_function_isa() -> ISA:
    """The widest ISA with tile functions the host executes, ``SCALAR`` (the ``omp simd`` lane loop) on any other."""
    supported = host_supported_isas()
    return next((isa for isa in FUNCTION_ISA_PREFERENCE if isa in supported), ISA.SCALAR)
