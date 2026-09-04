# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tile-op implementation-selection policy.

The tile lib nodes follow the standard DaCe expansion model: explicit per-node
``implementations`` (``'pure'`` scalar reference, ``'cutile'`` cuTile, and — as
they land — ``'avx512'`` / ``'avx2'`` / ``'sve'`` / ``'neon'`` CPU intrinsic
expansions). Selection is NOT an ``'Auto'`` re-dispatch (the
``CopyLibraryNode`` idiom for *expansion-time* context) — for tile nodes the
choice depends only on ``target_isa`` + ``K``, both known when the
``VectorizeCPUMultiDim`` orchestrator runs. So the orchestrator stamps
``node.target_isa`` and sets ``node.implementation = select_tile_implementation(
node)`` before ``expand_library_nodes()``.

Locked scope (2026-05-26): **K >= 2 -> ``'pure'``** (the two remainder knobs
ride on the scalar K-fold lowering); **K == 1 -> the per-ISA intrinsic** (or
``'pure'`` / ``'cutile'``). Until a node actually defines the real per-ISA
expansion class, the selector falls back to ``'pure'`` (the ISA name is not yet
in the node's ``implementations``), so wiring the orchestrator is safe before
the intrinsic headers exist.
"""
import functools
import platform

import dace
from dace.sdfg import nodes

# Implementation name each ``target_isa`` maps to for a K==1 tile (K>=2 is always
# ``'pure'``). Unknown / unset ISA falls back to ``'pure'``.
_ISA_TO_IMPL = {
    "AVX512": "avx512",
    "AVX2": "avx2",
    "ARM_SVE": "sve",
    "ARM_NEON": "neon",
    "CUDA": "cuda",
    "CUDA_WARP": "cuda_warp",
    "CUTILE": "cutile",
    # The K=1 scalar backend (dace/tile_ops/scalar.h call). Nodes without it yet
    # fall back to ``pure`` via the membership check in select_tile_implementation.
    "SCALAR": "scalar",
}

# The host-executed CPU SIMD ISAs. A tile op forced to one of these must run on a host that
# supports it (arch-native enforcement, see :func:`host_supported_isas`). SCALAR is always
# runnable; CUDA / CUTILE are GPU device ISAs gated by the schedule, not host-CPU-executed.
_CPU_SIMD_ISAS = frozenset({"AVX512", "AVX2", "ARM_SVE", "ARM_NEON"})

# TileBinop / TileUnop ops that have NO per-ISA single-char lowering and must use the
# ``pure`` loop expansion even at K=1 (a ``std::<fn>`` call the compiler's vector-math
# library / libmvec captures). These are the elemental math functions the frontend
# emits as bare calls: binary ``atan2`` / ``hypot`` / ``fmod`` and the extended trig
# ``tan`` / ``asin`` / ``acos`` / ``atan`` / ``sinh`` / ``cosh``; ``pow`` / ``ipow`` / ``**``
# for a true runtime exponent likewise has no ISA char (``ipow`` is the exact
# integer-exponent repeated-multiply ``dace::math::ipow``). (``sin`` / ``cos`` / ``exp`` /
# ``log`` / ``sqrt`` / ``tanh`` DO carry ISA char codes and stay on the intrinsic path.)
_PURE_ONLY_MATH_OPS = frozenset(
    {"atan2", "hypot", "fmod", "tan", "asin", "acos", "atan", "sinh", "cosh", "pow", "ipow", "**"})


@functools.lru_cache(maxsize=1)
def detect_host_isa() -> str:
    """Best K=1 tile-op ISA for the host, by CPU features (cached).

    The ``"AUTO"`` ``target_isa`` resolves through this at expansion time
    (overridable by passing an explicit ISA). x86: ``avx512f`` -> AVX512, else
    ``avx2`` -> AVX2. AArch64: SVE feature -> ARM_SVE, else ARM_NEON (baseline).
    Anything else falls back to ``SCALAR``.

    :returns: One of the :data:`_ISA_TO_IMPL` keys (never ``"AUTO"``).
    """
    machine = platform.machine().lower()
    flags = set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith(("flags", "Features")):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass
    if machine in ("x86_64", "amd64", "i386", "i686"):
        if "avx512f" in flags:
            return "AVX512"
        if "avx2" in flags:
            return "AVX2"
        return "SCALAR"
    if machine in ("aarch64", "arm64"):
        return "ARM_SVE" if "sve" in flags else "ARM_NEON"
    return "SCALAR"


@functools.lru_cache(maxsize=1)
def host_supported_isas() -> frozenset:
    """The set of K=1 tile-op ISAs the host can EXECUTE (cached).

    Arch-native enforcement: a forced ``target_isa`` outside this set would emit instructions the
    host cannot run (e.g. AVX-512 on an AVX2-only or ARM box) -- which still COMPILES under the
    backend's explicit ``-mavx512f`` but faults (SIGILL) at runtime. :func:`select_tile_implementation`
    refuses such an ISA early instead. Superset-closed: an AVX-512 host also runs AVX2 and SCALAR.
    ``SCALAR`` is always runnable; ``CUDA`` / ``CUTILE`` are GPU device ISAs (gated by the schedule),
    so they are absent here and the CPU-ISA guard skips them.

    Same host-feature source as :func:`detect_host_isa` (``/proc/cpuinfo``); where flags are
    unreadable (e.g. macOS has no ``/proc/cpuinfo``) it conservatively yields ``{SCALAR}``, matching
    ``detect_host_isa``'s fallback.

    :returns: Frozenset of :data:`_ISA_TO_IMPL` keys the host can execute (always includes ``SCALAR``).
    """
    machine = platform.machine().lower()
    flags = set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith(("flags", "Features")):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass
    if machine in ("x86_64", "amd64", "i386", "i686"):
        if "avx512f" in flags:
            return frozenset({"AVX512", "AVX2", "SCALAR"})
        if "avx2" in flags:
            return frozenset({"AVX2", "SCALAR"})
        return frozenset({"SCALAR"})
    if machine in ("aarch64", "arm64"):
        if "sve" in flags:
            return frozenset({"ARM_SVE", "ARM_NEON", "SCALAR"})
        return frozenset({"ARM_NEON", "SCALAR"})
    return frozenset({"SCALAR"})


def select_tile_implementation(node: nodes.LibraryNode, parent_state: dace.SDFGState = None) -> str:
    """Resolve the concrete tile-node implementation for the current target.

    :param node: The tile library node (carries ``widths`` and ``target_isa``).
    :param parent_state: Unused; kept for parity with the ``CopyLibraryNode``
        selector signature.
    :returns: A concrete implementation name present in ``node.implementations``.
        ``K >= 2`` is always ``'pure'``; ``K == 1`` maps ``target_isa`` through
        :data:`_ISA_TO_IMPL` (resolving ``"AUTO"`` to the host ISA via
        :func:`detect_host_isa`), falling back to ``'pure'`` when that per-ISA
        expansion is not yet defined on the node.
    """
    if len(node.widths) != 1:
        return "pure"
    # Elemental math ops with no per-ISA intrinsic lowering -- ``atan2`` / ``hypot`` /
    # ``fmod`` (binop) and the extended trig ``tan`` / ``asin`` / ``acos`` / ``atan`` /
    # ``sinh`` / ``cosh`` (unop), plus ``**`` / ``pow`` -- use the ``pure`` loop
    # expansion even at K=1: a per-lane ``std::<fn>`` call inside the tile for-loop
    # (under the ``_dace_tile_vectorize`` pragma) that the compiler's vector-math
    # library (libmvec) captures. The K=1 per-ISA backend has no single-char op code
    # for them (see ``_isa_codegen._OP_TO_CHAR`` / ``_UNOP_TO_CHAR``), so routing them
    # to the intrinsic path would ``KeyError``.
    if getattr(node, "op", None) in _PURE_ONLY_MATH_OPS:
        return "pure"
    target_isa = getattr(node, "target_isa", "SCALAR")
    if target_isa == "AUTO":
        target_isa = detect_host_isa()
    # Enforce arch-native: a forced CPU SIMD ISA the host cannot execute would emit a binary that
    # compiles (the backend adds its own ``-m`` flag) but SIGILLs at runtime -- the exact failure
    # seen when a test pins AVX-512 on an AVX2-only or ARM host. Refuse it early with a clear error
    # instead. ``AUTO`` already resolves to a host-supported ISA, so this only fires on an explicit
    # over-request; CUDA / CUTILE are GPU device ISAs and are not in ``_CPU_SIMD_ISAS``.
    if target_isa in _CPU_SIMD_ISAS and target_isa not in host_supported_isas():
        raise ValueError(f"tile-op target_isa={target_isa!r} is not executable on this host "
                         f"(supported: {sorted(host_supported_isas())}). Vectorization enforces "
                         f"arch-native: use ISA.AUTO to target the host, or pick a supported ISA.")
    # Complex operands have no packed-SIMD lowering on the CPU ISAs (add/sub are a trivial
    # interleaved real add, but mul needs shuffle/FCMLA sequences and abs/div have no SIMD
    # form), so route a complex tile op to the scalar ``pure`` loop over ``std::complex`` --
    # correct on every target, and the compiler may still auto-vectorize it. CUDA/CuTile
    # carries complex natively (``cuComplex``, scalar-per-lane warps) so it keeps its path.
    if target_isa != "CUDA" and _tile_has_complex_operand(node, parent_state):
        return "pure"
    impl = _ISA_TO_IMPL.get(target_isa, "pure")
    return impl if impl in node.implementations else "pure"


def _tile_has_complex_operand(node: nodes.LibraryNode, parent_state) -> bool:
    """True iff any connected operand / output of the tile op is a complex dtype.

    ``parent_state`` is the containing :class:`~dace.sdfg.state.SDFGState`; ``None``
    (unknown context) conservatively returns False (keep the ISA path).
    """
    if parent_state is None:
        return False
    sdfg = parent_state.sdfg
    for e in (*parent_state.in_edges(node), *parent_state.out_edges(node)):
        if e.data is None or e.data.data is None:
            continue
        desc = sdfg.arrays.get(e.data.data)
        if desc is not None and desc.dtype in (dace.dtypes.complex64, dace.dtypes.complex128):
            return True
    return False
