# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared harness for the ``simplify -> loop2map -> mapfusion`` then multi-dim
tile-op vectorization corpus (npbench + polybench).

The two corpus test files (``npbench_simplify_multidim_vectorize_corpus_test.py``
and ``polybench_simplify_multidim_vectorize_corpus_test.py``) share:

* :func:`base_pipeline` -- the light base transform (``simplify`` +
  ``LoopToMap`` + ``MapFusion``) applied to a fresh SDFG. The base SDFG is the
  numerically-checkable starting point AND the shared root every vectorize
  config deep-copies.
* :func:`select_widths` -- per-kernel tile widths: ``(8, 8)`` when every
  innermost map has >= 2 params, else ``(8,)`` (mixed-K within one SDFG is
  unsupported by the tile pipeline, so any 1-D map pins the whole SDFG to K=1).
* :data:`CONFIGS` / :data:`PHASES` -- the 4 vectorize configs
  (``{AVX512, SCALAR} x {merge, fp_factor}``, all ``scalar_postamble``
  remainder) plus the ``base`` (no-vectorize) phase.
* :func:`make_pass` -- build the :class:`VectorizeCPUMultiDim` for one config.

Each corpus file supplies its own loader (inputs / reference / run / compare)
because npbench (numpy oracle) and polybench (value-preservation vs the
untransformed baseline) differ.
"""
from typing import Dict, Tuple

from dace.libraries.tileops._dispatch import detect_host_isa
from dace.sdfg import nodes as nd
from dace.transformation.dataflow import MapFusionHorizontal, MapFusionVertical
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

#: Vectorize configs = the cross product of {host-native ISA, SCALAR} x {merge, fp_factor} branch
#: mode, all ``remainder_strategy="scalar_postamble"`` per the corpus spec ("merge" = per-lane
#: ``TileITE`` select; "fp_factor" = ``c*x + (1-c)*y`` arithmetic). The SIMD ISA is the HOST's
#: (``detect_host_isa`` -> AVX512 / AVX2 / ARM_SVE / ARM_NEON / SCALAR), NOT a hardcoded AVX-512:
#: vectorization enforces arch-native (a forced non-host ISA would SIGILL at runtime -- see
#: ``_dispatch.host_supported_isas``), so pinning AVX-512 made every avx512 phase fail on an
#: AVX2-only or ARM box. Host-native keeps full SIMD-vs-scalar coverage on every machine with no
#: skips. When the host is itself SCALAR the two blocks collapse (dict dedupes the equal keys).
_HOST_ISA = detect_host_isa()
CONFIGS: Dict[str, dict] = {
    f"{isa.lower()}_{short}": dict(target_isa=isa, branch_mode=mode)
    for isa in (_HOST_ISA, "SCALAR")
    for short, mode in (("merge", "merge"), ("fpfac", "fp_factor"))
}

#: Parametrized phases: the base (no-vectorize) numerical check plus one per
#: vectorize config. ``base`` must pass for a vectorize config to be meaningful.
PHASES: Tuple[str, ...] = ("base", *CONFIGS)


def base_pipeline(sdfg) -> None:
    """Apply the light base transform in place: ``simplify`` -> ``LoopToMap`` ->
    ``MapFusion`` (vertical + horizontal) -> ``simplify``.

    Every step runs with ``validate=True, validate_all=True`` so a
    transformation that malforms the SDFG is caught at its source (deep
    per-application validation), matching the corpus's ``validate_all`` posture.

    This is the corpus's base pipeline: it turns data-parallel loops into maps
    and fuses adjacent maps, leaving a maps-based SDFG the tile vectorizer
    consumes. It is deliberately lighter than ``canonicalize`` (no LICM,
    interchange, reduction/einsum lift, scan detection, ...); the vectorizer
    integrates the reduction/einsum lifting it needs internally.
    """
    sdfg.simplify(validate=True, validate_all=True)
    sdfg.apply_transformations_repeated(LoopToMap, permissive=False, validate=True, validate_all=True)
    sdfg.apply_transformations_repeated([MapFusionVertical, MapFusionHorizontal],
                                        permissive=False,
                                        validate=True,
                                        validate_all=True)
    sdfg.simplify(validate=True, validate_all=True)


def select_widths(sdfg) -> Tuple[int, ...]:
    """Per-kernel tile widths: ``(8, 8)`` if every innermost map carries >= 2
    params, else ``(8,)``.

    The tile pipeline requires a uniform K across the whole SDFG (mixed-K
    aborts), so a single 1-D map (an init / reduction / boundary map beside a
    2-D body) pins the entire SDFG to K=1. Mirrors the width-selection rule in
    ``tsvc_canonicalize_vectorize_corpus_test.py``.
    """
    counts = [len(n.map.params) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry)]
    return (8, 8) if (counts and min(counts) >= 2) else (8, )


def make_pass(widths: Tuple[int, ...], config: str) -> VectorizeCPUMultiDim:
    """Build the :class:`VectorizeCPUMultiDim` for one named config in
    :data:`CONFIGS`, at the given ``widths``.

    All configs pin ``remainder_strategy="scalar_postamble"`` and
    ``validate_all=True`` (the corpus spec); the config name selects the ISA
    and branch mode.
    """
    return VectorizeCPUMultiDim(
        VectorizeConfig(widths=widths,
                        remainder_strategy=RemainderStrategy.SCALAR_POSTAMBLE,
                        validate_all=True,
                        **CONFIGS[config]))
