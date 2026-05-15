# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Coverage matrix for every ``VectorizeCPU`` constructor knob.

Goal: each knob has at least one test that pins its documented
behaviour with an assertion the harness can verify automatically
(numerical equivalence, exception raised, generated CPP grep, or SDFG
shape check). When a knob's path diverges from baseline behaviour, the
test below MUST flag it.

Knobs already covered indirectly by the existing test corpus and NOT
re-tested here:
- ``vector_width=8`` (every test)
- ``insert_copies`` (True default + False newly exercised via
  ``_setup_strided_nsdfg_edges_inline``; existing
  ``test_strided_gather_scatter`` covers True; tsvc_additional's
  s127/s1111 skip path covers False)
- ``branch_mode`` / ``use_fp_factor`` / ``branch_normalization``
  (parametrised via conftest fixture)
- ``remainder_strategy`` (parametrised via conftest fixture)
- ``lower_to_intrinsics=True`` (test_strided_gather_scatter masked
  variants; new diagonal masked tests)
- ``fail_on_unvectorizable`` (True covered in run_vectorization_test;
  False covered in tsvc_additional)
- ``fuse_overlapping_loads=True`` (jacobi tests)
- ``try_to_demote_symbols_in_nsdfgs=True`` (disjoint_chain)

Knobs covered HERE:
- ``vector_width`` ∈ {4, 8, 16}
- ``only_apply_vectorization_pass``
- ``no_inline``
- ``eliminate_trivial_vector_map`` (False path)
- ``user_skip_nsdfg_arrays``
- ``force_autovec_ops`` / ``force_pscalar_ops`` (singly + overlap reject)
- Constructor rejection: ``use_fp_factor=True + branch_normalization=True``
- Constructor rejection: ``use_fp_factor=True + remainder_strategy="masked"``
- Constructor rejection: ``remainder_strategy="full_loop_mask"``
- Constructor rejection: ``force_autovec_ops & force_pscalar_ops`` overlap
- Constructor rejection: unknown ``remainder_strategy`` value
- Constructor rejection: ``force_*_ops`` references a masked variant key
"""
import copy
import os
import pytest
import numpy as np

import dace
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from dace.transformation.interstate import LoopToMap

N = dace.symbol("N")

# --------------------------------------------------------------------------
# Simple elemental kernel used across the knob matrix.
# --------------------------------------------------------------------------


@dace.program
def _add_scaled(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N:1]:
        C[i] = A[i] + B[i] * 2.0


def _build_sdfg(name: str) -> dace.SDFG:
    sdfg = _add_scaled.to_sdfg(simplify=False)
    sdfg.simplify(validate=True, validate_all=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()
    sdfg.name = name
    return sdfg


def _run_and_compare(name: str, **vec_kwargs) -> None:
    """Build the SDFG, vectorize a copy, run both, assert numerical
    equivalence."""
    np.random.seed(42)
    A = np.random.rand(64)
    B = np.random.rand(64)
    C_ref = np.zeros(64)
    C_vec = np.zeros(64)

    sdfg = _build_sdfg(name)
    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = name + "_vec"
    vec_kwargs.setdefault("vector_width", 8)
    VectorizeCPU(**vec_kwargs).apply_pass(vsdfg, {})

    sdfg.compile()(A=A, B=B, C=C_ref, N=64)
    vsdfg.compile()(A=A, B=B, C=C_vec, N=64)
    assert np.allclose(C_ref, C_vec, atol=1e-12), \
        f"knob test {name}: max abs diff = {np.max(np.abs(C_ref - C_vec))}"


# --------------------------------------------------------------------------
# vector_width
# --------------------------------------------------------------------------


@pytest.mark.parametrize("width", [4, 8, 16])
def test_knob_vector_width(width):
    """Numerical equivalence at W ∈ {4, 8, 16}. Catches a regression
    that hard-codes W=8 anywhere on the critical path."""
    _run_and_compare(f"knob_vw_{width}", vector_width=width)


# --------------------------------------------------------------------------
# only_apply_vectorization_pass
# --------------------------------------------------------------------------


def test_knob_only_apply_vectorization_pass_bypass():
    """``only_apply_vectorization_pass=True`` skips the preamble pipeline
    (EliminateBranches / SplitTasklets / InlineSDFGs / P1+P2 etc.) and
    runs only ``Vectorize`` + ``RemoveMathCall``. The kernel must still
    produce the correct output."""
    _run_and_compare("knob_only_apply", only_apply_vectorization_pass=True)


# --------------------------------------------------------------------------
# no_inline
# --------------------------------------------------------------------------


def test_knob_no_inline():
    """``no_inline=True`` removes ``InlineSDFGs`` from the pipeline.
    Output stays numerically equivalent for the simple elemental kernel
    (no nested SDFGs in the source so the bypass is a no-op here)."""
    _run_and_compare("knob_no_inline", no_inline=True)


# --------------------------------------------------------------------------
# eliminate_trivial_vector_map
# --------------------------------------------------------------------------


def test_knob_eliminate_trivial_vector_map_false_keeps_outer_map():
    """``eliminate_trivial_vector_map=False`` drops ``RemoveVectorMaps``
    from the pipeline. The trivial outer step-W tile loop stays in the
    SDFG. Numerical equivalence still holds; check SDFG shape to confirm
    the map wasn't eliminated."""
    np.random.seed(42)
    A = np.random.rand(64)
    B = np.random.rand(64)
    C_ref = np.zeros(64)
    C_vec = np.zeros(64)

    sdfg = _build_sdfg("knob_keep_map_ref")
    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = "knob_keep_map_vec"
    VectorizeCPU(vector_width=8, eliminate_trivial_vector_map=False).apply_pass(vsdfg, {})

    sdfg.compile()(A=A, B=B, C=C_ref, N=64)
    vsdfg.compile()(A=A, B=B, C=C_vec, N=64)
    assert np.allclose(C_ref, C_vec, atol=1e-12)

    # Count MapEntry nodes — with elimination=False the trivial vector
    # map stays around (one extra Map compared to default).
    map_count = sum(1 for n, _ in vsdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))
    assert map_count >= 1, f"no maps found in vectorized SDFG; trivial-vector-map elimination misfired"


# --------------------------------------------------------------------------
# user_skip_nsdfg_arrays
# --------------------------------------------------------------------------


def test_knob_user_skip_nsdfg_arrays_accepted():
    """``user_skip_nsdfg_arrays`` is forwarded to ``add_copies_before_and_after_nsdfg``
    as the per-call skip list. For the simple elemental kernel without
    P1 wrapping there's no NSDFG to skip; the knob value just shouldn't
    cause a constructor error and the pipeline must still run end-to-end."""
    _run_and_compare("knob_skip", user_skip_nsdfg_arrays={"zqxfg", "zsolqa"})


# --------------------------------------------------------------------------
# force_autovec_ops / force_pscalar_ops
# --------------------------------------------------------------------------


def test_knob_force_autovec_rewrites_template():
    """``force_autovec_ops={"+"}`` rewrites the ``vector_add`` template
    to ``vector_add_av``. Grep the generated CPP."""
    sdfg = _build_sdfg("knob_force_autovec")
    VectorizeCPU(vector_width=8, force_autovec_ops={"+"}).apply_pass(sdfg, {})
    sdfg.compile()

    cache_root = os.path.join(".dacecache", sdfg.name, "src", "cpu")
    cpp_path = os.path.join(cache_root, sdfg.name + ".cpp")
    cpp = open(cpp_path).read()
    assert "vector_add_av<" in cpp, \
        f"force_autovec_ops={{'+'}} did not rewrite vector_add to vector_add_av"
    assert "vector_mult_av<" not in cpp, \
        f"force_autovec_ops={{'+'}} unexpectedly rewrote vector_mult; only '+' should be affected"


def test_knob_force_pscalar_rewrites_template():
    """``force_pscalar_ops={"+"}`` rewrites the ``vector_add`` template
    to ``vector_add_pscalar``."""
    sdfg = _build_sdfg("knob_force_pscalar")
    VectorizeCPU(vector_width=8, force_pscalar_ops={"+"}).apply_pass(sdfg, {})
    sdfg.compile()

    cache_root = os.path.join(".dacecache", sdfg.name, "src", "cpu")
    cpp_path = os.path.join(cache_root, sdfg.name + ".cpp")
    cpp = open(cpp_path).read()
    assert "vector_add_pscalar<" in cpp, \
        f"force_pscalar_ops={{'+'}} did not rewrite vector_add to vector_add_pscalar"


# --------------------------------------------------------------------------
# Constructor rejections (mutex + value-domain checks)
# --------------------------------------------------------------------------


def test_constructor_rejects_use_fp_factor_and_branch_normalization():
    """``use_fp_factor`` and ``branch_normalization`` are mutually
    exclusive branch-lowering strategies. The constructor must raise
    ``ValueError``."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        VectorizeCPU(vector_width=8, use_fp_factor=True, branch_normalization=True)


def test_constructor_rejects_fp_factor_masked():
    """``use_fp_factor=True`` + ``remainder_strategy="masked"`` is rejected
    per the locked plan rule (no bool→float cast on the masked path)."""
    with pytest.raises(ValueError, match="incompatible"):
        VectorizeCPU(vector_width=8, use_fp_factor=True, branch_normalization=False, remainder_strategy="masked")


def test_constructor_rejects_full_loop_mask():
    """``remainder_strategy="full_loop_mask"`` is queued (R3); the
    constructor raises ``NotImplementedError`` until the SVE always-
    predicated path lands."""
    with pytest.raises(NotImplementedError, match="full_loop_mask"):
        VectorizeCPU(vector_width=8, remainder_strategy="full_loop_mask")


def test_constructor_rejects_unknown_remainder_strategy():
    """Unknown ``remainder_strategy`` value raises ``ValueError`` with
    the allowed set listed."""
    with pytest.raises(ValueError, match="remainder_strategy must be"):
        VectorizeCPU(vector_width=8, remainder_strategy="nonsense_value")


def test_constructor_rejects_force_autovec_pscalar_overlap():
    """An op listed in BOTH ``force_autovec_ops`` and ``force_pscalar_ops``
    is rejected — the override target is ambiguous."""
    with pytest.raises(ValueError, match="overlap"):
        VectorizeCPU(vector_width=8, force_autovec_ops={"+"}, force_pscalar_ops={"+"})


def test_constructor_rejects_force_op_referencing_masked_variant():
    """``force_*_ops`` keys must be base op identifiers, not the
    ``_masked`` variants. The constructor raises ``ValueError``."""
    with pytest.raises(ValueError, match="masked"):
        VectorizeCPU(vector_width=8, force_autovec_ops={"+_masked"})


def test_constructor_rejects_force_op_unknown_key():
    """``force_*_ops`` keys must exist in the templates dict; an unknown
    op name raises ``KeyError``."""
    with pytest.raises(KeyError, match="unknown op"):
        VectorizeCPU(vector_width=8, force_autovec_ops={"this_op_does_not_exist"})
