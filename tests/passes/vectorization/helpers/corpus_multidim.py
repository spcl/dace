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
* :func:`phases_for` -- the phases to run for ONE kernel: the ``fp_factor``
  configs are generated only for kernels that carry a conditional (see
  :func:`exercises_branch_lowering`).
* :func:`make_pass` -- build the :class:`VectorizeCPUMultiDim` for one config.

Each corpus file supplies its own loader (inputs / reference / run / compare)
because npbench (numpy oracle) and polybench (value-preservation vs the
untransformed baseline) differ.
"""
import ast
import inspect
import textwrap
from typing import Dict, Tuple

from dace.frontend.python.parser import DaceProgram
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
#: AVX2-only or ARM box.
#:
#: The arm is labelled ``hostsimd``, NOT by the detected ISA: the label reaches the pytest test ID,
#: and an ISA-derived one renamed every SIMD case per runner (``[...-avx512_merge]`` vs
#: ``[...-avx2_merge]``), so a before/after ID diff across a heterogeneous runner pool showed a
#: full rename instead of the real delta. The ISA still varies per host -- it just stops leaking
#: into the identifier. On a SCALAR-only host the two arms are the same config; both still run (the
#: ID set must not depend on the runner's CPU either).
_HOST_ISA = detect_host_isa()
CONFIGS: Dict[str, dict] = {
    f"{label}_{short}": dict(target_isa=isa, branch_mode=mode)
    for label, isa in (("hostsimd", _HOST_ISA), ("scalar", "SCALAR"))
    for short, mode in (("merge", "merge"), ("fpfac", "fp_factor"))
}

#: Parametrized phases: the base (no-vectorize) numerical check plus one per
#: vectorize config. ``base`` must pass for a vectorize config to be meaningful.
PHASES: Tuple[str, ...] = ("base", *CONFIGS)

#: Calls that can lower to a conditional / select. Over-broad on purpose (see
#: :func:`exercises_branch_lowering`).
_BRANCH_CALLS = frozenset({"where", "select", "clip", "min", "max", "minimum", "maximum", "fmin", "fmax"})


def exercises_branch_lowering(program) -> bool:
    """Can ``branch_mode`` change this kernel's lowering at all?

    ``fp_factor`` differs from ``merge`` by exactly three extra passes:
    :class:`LowerITEToFpFactor` (rewrites an INTEGER-dtype ``ITE(c, t, e)`` tasklet body to
    ``c*t + (1-c)*e``), :class:`EliminateBranches` (matches a ``ConditionalBlock``) and
    :class:`LowerInterstateConditionalAssignmentsToTasklets` (consumes only the
    ``condition_symbol_to_scalar`` tasklets ``EliminateBranches`` mints). Every one of them needs
    a conditional in the SDFG, and neither the python frontend nor :func:`base_pipeline`
    (``simplify`` / ``LoopToMap`` / ``MapFusion``) manufactures one from straight-line source --
    so on a branchless kernel both modes run the identical pipeline and emit identical code.

    Read off the kernel source, never a name list, so a kernel that GAINS an ``if/else`` picks the
    ``fp_factor`` phase back up on its own. Deliberately over-broad -- ``Compare`` / ``BoolOp`` /
    ``min`` / ``max`` all count and unreadable source answers ``True`` -- because a false positive
    costs one redundant phase while a false negative would drop coverage.

    :param program: the kernel's ``@dace.program``.
    """
    return _branches(program.f, set())


def _branches(fn, seen) -> bool:
    """``exercises_branch_lowering`` over one function, recursing into the ``@dace.program`` and
    module-local helpers it calls -- npbench kernels push their conditionals into helpers
    (``nbody`` -> ``getAcc``), which a source scan of the entry alone would miss."""
    if fn in seen:
        return False
    seen.add(fn)
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except Exception:  # noqa: BLE001 -- cannot read the source; assume it branches
        return True
    # ``getsource`` includes the decorators; they are registration metadata, not kernel code, and
    # recursing into one (``@tsvc_kernel``) would mark every kernel as branching.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    glb = fn.__globals__
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.BoolOp, ast.Compare)):
            return True
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
        # ``np.divide(..., where=mask)`` is a predicated write, same as an if/else.
        if name in _BRANCH_CALLS or any(kw.arg == "where" for kw in node.keywords):
            return True
        callee = glb.get(name)
        callee = callee.f if isinstance(callee, DaceProgram) else callee
        if inspect.isfunction(callee) and _branches(callee, seen):
            return True
    return False


def phases_for(program) -> Tuple[str, ...]:
    """Phases to run for ONE kernel: :data:`PHASES` minus the ``fp_factor`` configs when the
    kernel carries no conditional for them to act on (see :func:`exercises_branch_lowering`)."""
    if exercises_branch_lowering(program):
        return PHASES
    return tuple(p for p in PHASES if not p.endswith("_fpfac"))


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
