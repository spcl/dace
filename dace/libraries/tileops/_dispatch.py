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
    "CUTILE": "cutile",
    # The K=1 scalar backend (dace/tile_ops/scalar.h call). Nodes without it yet
    # fall back to ``pure`` via the membership check in select_tile_implementation.
    "SCALAR": "scalar",
}


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
    target_isa = getattr(node, "target_isa", "SCALAR")
    if target_isa == "AUTO":
        target_isa = detect_host_isa()
    impl = _ISA_TO_IMPL.get(target_isa, "pure")
    return impl if impl in node.implementations else "pure"
