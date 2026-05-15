# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for kernels whose iteration count is NOT divisible by vector_width.

Kernels are written with plain Python ``for i in range(...)`` loops
(NOT ``dace.map[...]``). LoopToMap is applied before VectorizeCPU to
convert the loop into a Map; this matches the typical front-end flow
(TSVC tests follow the same pattern).

There is no ``"divides_evenly"`` strategy: P2
(``SplitMapForVectorRemainder``) always runs and decides for itself
whether a remainder is needed via symbolic divisibility analysis —
when ``simplify(ub-lb+1) % W == 0`` is provably true (a constant
multiple of W, or a symbolic ``8*N``) it skips the split entirely.
``remainder_strategy`` only selects the remainder *shape* when the
split does happen:

- ``"scalar"`` (R1): main step-W map + a step-1 sequential scalar
  postamble. No mask. Robust for any N.
- ``"masked"`` (R2): main step-W (no mask) + step-W remainder body
  with a P3 ``_iter_mask``.
- ``"full_loop_mask"``: queued (R3); raises ``NotImplementedError``.
"""
import copy
import math

import dace
import numpy as np
import pytest

from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU


N = dace.symbol("N")


@dace.program
def shift_plus_one(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1):
        a[i] = 2.0 * b[i + 1]


@dace.program
def shift_plus_two(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 2):
        a[i] = 3.0 * b[i + 2]


# Additional kernels exercising remainder under different op shapes —
# add (vec+vec), mul (vec*scalar), elementwise sqrt (unary math),
# min/max (binary library calls).


@dace.program
def add_vec_vec(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
    for i in range(N):
        out[i] = a[i] + b[i]


@dace.program
def mul_vec_scalar(a: dace.float64[N], out: dace.float64[N]):
    for i in range(N):
        out[i] = a[i] * 3.5


@dace.program
def sqrt_unary(a: dace.float64[N], out: dace.float64[N]):
    for i in range(N):
        out[i] = math.sqrt(a[i])


@dace.program
def min_vec_vec(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
    for i in range(N):
        out[i] = min(a[i], b[i])


@dace.program
def fused_chain(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], out: dace.float64[N]):
    """Multi-op kernel: out[i] = (a[i] + b[i]) * c[i] + 1.5"""
    for i in range(N):
        out[i] = (a[i] + b[i]) * c[i] + 1.5


@pytest.fixture(params=["scalar", "masked"])
def remainder_strategy(request):
    """Parametrise tests across the currently-wired remainder strategies."""
    return request.param


def _branch_kwargs(remainder_strategy: str) -> dict:
    """fp_factor is incompatible with the masked remainder (locked rule);
    these kernels are branchless so branch_normalization is equivalent."""
    if remainder_strategy == "masked":
        return dict(use_fp_factor=False, branch_normalization=True)
    return dict(use_fp_factor=True, branch_normalization=False)


def _run(prog, Nv: int, remainder_strategy: str):
    """Build SDFG with LoopToMap, apply VectorizeCPU, compare to scalar reference."""
    a_ref = np.zeros(Nv)
    b = np.random.rand(Nv)
    a_vec = np.zeros(Nv)

    sdfg = prog.to_sdfg(simplify=True)
    sdfg.name = f"{prog.name}_{Nv}_{remainder_strategy}_ref"
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"{prog.name}_{Nv}_{remainder_strategy}_v"

    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True,
                 insert_copies=False,
                 remainder_strategy=remainder_strategy,
                 **_branch_kwargs(remainder_strategy)).apply_pass(vsdfg, {})

    sdfg(a=a_ref, b=b, N=Nv)
    vsdfg(a=a_vec, b=b, N=Nv)
    diff = np.max(np.abs(a_ref - a_vec))
    return a_ref, a_vec, diff


# Non-divisible iteration counts (force remainder/mask handling).
# Pinned to ["scalar"] — P2 splits these (divisibility cannot be
# proven) and the scalar postamble is the robust shape for any N.


@pytest.mark.parametrize("remainder_strategy", ["scalar"])
def test_shift_plus_one_n10_remainder(remainder_strategy):
    """N=10, range(N-1)=range(9) = 9 iters → 1 vector tile + 1 remainder."""
    a_ref, a_vec, diff = _run(shift_plus_one, Nv=10, remainder_strategy=remainder_strategy)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


@pytest.mark.parametrize("remainder_strategy", ["scalar"])
def test_shift_plus_one_n15_remainder(remainder_strategy):
    """N=15, range(N-1)=range(14) = 14 iters → 1 vector + 6 remainder."""
    a_ref, a_vec, diff = _run(shift_plus_one, Nv=15, remainder_strategy=remainder_strategy)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


@pytest.mark.parametrize("remainder_strategy", ["scalar"])
def test_shift_plus_two_n11_remainder(remainder_strategy):
    """+2 shift, N=11, range(N-2)=range(9) = 9 iters → 1 vector + 1 remainder."""
    a_ref, a_vec, diff = _run(shift_plus_two, Nv=11, remainder_strategy=remainder_strategy)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


@pytest.mark.parametrize("remainder_strategy", ["scalar"])
def test_shift_plus_two_n15_remainder(remainder_strategy):
    """+2 shift, N=15, range(N-2)=range(13) = 13 iters → 1 vector + 5 remainder."""
    a_ref, a_vec, diff = _run(shift_plus_two, Nv=15, remainder_strategy=remainder_strategy)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


# Divides-evenly sanity tests (regression guards on the working path).


def test_shift_plus_one_n9_divides_evenly(remainder_strategy):
    """N=9, range(N-1)=range(8) = 8 iters, divisible by 8."""
    a_ref, a_vec, diff = _run(shift_plus_one, Nv=9, remainder_strategy=remainder_strategy)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


def test_shift_plus_two_n10_divides_evenly(remainder_strategy):
    """N=10, range(N-2)=range(8) = 8 iters, divisible by 8."""
    a_ref, a_vec, diff = _run(shift_plus_two, Nv=10, remainder_strategy=remainder_strategy)
    assert diff < 1e-12, f"max abs diff = {diff}\nref={a_ref}\nvec={a_vec}"


# Additional kernel shapes at non-divisible N (forces remainder handling).


def _run3(prog, Nv: int, remainder_strategy: str, in_arrays: list, out_arrays: list):
    """Like ``_run`` but accepts arbitrary in/out array lists by name."""
    rng = np.random.default_rng(seed=Nv)
    arrs_ref = {k: rng.random(Nv) for k in in_arrays}
    arrs_ref.update({k: np.zeros(Nv) for k in out_arrays})
    arrs_vec = {k: arrs_ref[k].copy() for k in arrs_ref}

    sdfg = prog.to_sdfg(simplify=True)
    sdfg.name = f"{prog.name}_{Nv}_{remainder_strategy}_ref"
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"{prog.name}_{Nv}_{remainder_strategy}_v"

    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True,
                 insert_copies=False,
                 remainder_strategy=remainder_strategy,
                 **_branch_kwargs(remainder_strategy)).apply_pass(vsdfg, {})

    sdfg(**arrs_ref, N=Nv)
    vsdfg(**arrs_vec, N=Nv)
    diffs = {k: np.max(np.abs(arrs_ref[k] - arrs_vec[k])) for k in out_arrays}
    return arrs_ref, arrs_vec, diffs


# For non-divisible N these pin remainder_strategy="scalar" explicitly
# (the robust step-1 postamble shape).


@pytest.mark.parametrize("Nv", [7, 9, 15, 17])
def test_add_vec_vec_nondivisible_scalar(Nv: int):
    """out[i] = a[i] + b[i] over N iterations, N not divisible by W=8."""
    ref, vec, diffs = _run3(add_vec_vec, Nv, "scalar", ["a", "b"], ["out"])
    assert diffs["out"] < 1e-12, f"diffs={diffs}"


@pytest.mark.parametrize("Nv", [7, 9, 15, 17])
def test_mul_vec_scalar_nondivisible_scalar(Nv: int):
    """out[i] = a[i] * 3.5 over N iterations, N not divisible by W=8."""
    ref, vec, diffs = _run3(mul_vec_scalar, Nv, "scalar", ["a"], ["out"])
    assert diffs["out"] < 1e-12, f"diffs={diffs}"


@pytest.mark.parametrize("Nv", [7, 9, 15, 17])
def test_sqrt_unary_nondivisible_scalar(Nv: int):
    """out[i] = sqrt(a[i]) over N iterations, N not divisible by W=8."""
    ref, vec, diffs = _run3(sqrt_unary, Nv, "scalar", ["a"], ["out"])
    assert diffs["out"] < 1e-12, f"diffs={diffs}"


@pytest.mark.parametrize("Nv", [7, 9, 15, 17])
def test_min_vec_vec_nondivisible_scalar(Nv: int):
    """out[i] = min(a[i], b[i]) over N iterations, N not divisible by W=8."""
    ref, vec, diffs = _run3(min_vec_vec, Nv, "scalar", ["a", "b"], ["out"])
    assert diffs["out"] < 1e-12, f"diffs={diffs}"


@pytest.mark.parametrize("Nv", [7, 9, 15, 17])
def test_fused_chain_nondivisible_scalar(Nv: int):
    """out[i] = (a[i] + b[i]) * c[i] + 1.5 over N iterations, N not divisible by W=8."""
    ref, vec, diffs = _run3(fused_chain, Nv, "scalar", ["a", "b", "c"], ["out"])
    assert diffs["out"] < 1e-12, f"diffs={diffs}"


# Constant map range: when the iteration count is a known constant divisible
# by W, P2 (SplitMapForVectorRemainder) should detect it and SKIP the split
# (`trip %% W == 0` evaluates to True statically). The divides_evenly path
# stays safe because the range is genuinely divisible.


@dace.program
def constant_range_div_16(a: dace.float64[16], out: dace.float64[16]):
    """N=16 constant, divisible by W=8. Should fully vectorize cleanly."""
    for i in range(16):
        out[i] = a[i] * 2.5


@dace.program
def constant_range_div_24(a: dace.float64[24], out: dace.float64[24]):
    """N=24 constant, divisible by W=8. Three full tiles, no remainder."""
    for i in range(24):
        out[i] = a[i] + 1.0


def test_constant_range_div_16(remainder_strategy):
    """Constant range divisible by W. Both strategies must produce correct
    output; ``scalar`` should be a no-op split (P2 detects divisibility)."""
    rng = np.random.default_rng(seed=16)
    a = rng.random(16)
    out_ref = np.zeros(16)
    out_vec = np.zeros(16)

    sdfg = constant_range_div_16.to_sdfg(simplify=True)
    sdfg.name = f"constant_range_div_16_{remainder_strategy}_ref"
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"constant_range_div_16_{remainder_strategy}_v"
    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True,
                 insert_copies=False,
                 remainder_strategy=remainder_strategy,
                 **_branch_kwargs(remainder_strategy)).apply_pass(vsdfg, {})

    sdfg(a=a, out=out_ref)
    vsdfg(a=a, out=out_vec)
    assert np.max(np.abs(out_ref - out_vec)) < 1e-12


def test_constant_range_div_24(remainder_strategy):
    """Constant range divisible by W. 3 full tiles, P2 skip on scalar mode."""
    rng = np.random.default_rng(seed=24)
    a = rng.random(24)
    out_ref = np.zeros(24)
    out_vec = np.zeros(24)

    sdfg = constant_range_div_24.to_sdfg(simplify=True)
    sdfg.name = f"constant_range_div_24_{remainder_strategy}_ref"
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"constant_range_div_24_{remainder_strategy}_v"
    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True,
                 insert_copies=False,
                 remainder_strategy=remainder_strategy,
                 **_branch_kwargs(remainder_strategy)).apply_pass(vsdfg, {})

    sdfg(a=a, out=out_ref)
    vsdfg(a=a, out=out_vec)
    assert np.max(np.abs(out_ref - out_vec)) < 1e-12


# Provable symbolic divisibility: a map over ``0 : 8*M`` has a trip
# count that is a symbolic multiple of W=8, so P2
# (SplitMapForVectorRemainder) must prove ``trip % 8 == 0`` and SKIP
# the split — the resulting SDFG carries NO remainder map.  This is the
# replacement for the old explicit ``divides_evenly`` mode: divisibility
# is detected, not declared by the caller.


_SYMBOLIC_DIV_M = dace.symbol("M")


@dace.program
def symbolic_mult_of_w(a: dace.float64[8 * _SYMBOLIC_DIV_M], out: dace.float64[8 * _SYMBOLIC_DIV_M]):
    """Trip count ``8*M`` — provably divisible by W=8 for any M."""
    for i in range(8 * _SYMBOLIC_DIV_M):
        out[i] = a[i] * 2.0


def _count_remainder_maps(sdfg: dace.SDFG) -> int:
    """Number of P2-emitted remainder maps in ``sdfg``.

    P2's scalar remainder is a ``ScheduleType.Sequential`` map; its
    masked remainder carries a ``__masked_rem`` label suffix.  Either
    marks a map that only exists because the split fired.

    :param sdfg: SDFG to scan.
    :returns: Count of remainder maps (0 when P2 skipped the split).
    """
    n = 0
    for node, _ in sdfg.all_nodes_recursive():
        if not isinstance(node, dace.nodes.MapEntry):
            continue
        if node.map.schedule == dace.dtypes.ScheduleType.Sequential:
            n += 1
        elif node.map.label.endswith("__masked_rem"):
            n += 1
    return n


def test_symbolic_8m_p2_skips_split():
    """P2 alone: a ``0 : 8*M`` map proves divisibility and does NOT split.

    ``apply_pass`` returns ``None`` (zero applications) and the map
    count is unchanged — no remainder map is introduced.
    """
    from dace.transformation.passes.vectorization.nest_innermost_map_body import NestInnermostMapBodyIntoNSDFG
    from dace.transformation.passes.vectorization.split_map_for_vector_remainder import SplitMapForVectorRemainder

    sdfg = symbolic_mult_of_w.to_sdfg(simplify=True)
    sdfg.name = "symbolic_8m_p2_skip"
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})

    maps_before = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))
    applied = SplitMapForVectorRemainder(vector_width=8, mode="scalar").apply_pass(sdfg, {})
    maps_after = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))

    assert applied is None, f"P2 split a provably-divisible 8*M map (applied={applied})"
    assert maps_after == maps_before, f"map count changed {maps_before} -> {maps_after} (a remainder was emitted)"
    assert _count_remainder_maps(sdfg) == 0, "a remainder map exists for a provably-divisible 8*M trip"


def test_symbolic_8m_no_remainder_end_to_end(remainder_strategy):
    """Full VectorizeCPU on a ``0 : 8*M`` kernel: no remainder map, and
    the vectorized output matches the scalar reference for several M."""
    sdfg = symbolic_mult_of_w.to_sdfg(simplify=True)
    sdfg.name = f"symbolic_8m_e2e_{remainder_strategy}_ref"
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"symbolic_8m_e2e_{remainder_strategy}_v"
    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True,
                 insert_copies=False,
                 remainder_strategy=remainder_strategy,
                 **_branch_kwargs(remainder_strategy)).apply_pass(vsdfg, {})

    assert _count_remainder_maps(vsdfg) == 0, (
        f"VectorizeCPU emitted a remainder map for a provably-divisible "
        f"8*M trip (strategy={remainder_strategy})")

    for Mv in (1, 2, 5):
        Nv = 8 * Mv
        rng = np.random.default_rng(seed=Nv)
        a = rng.random(Nv)
        out_ref = np.zeros(Nv)
        out_vec = np.zeros(Nv)
        sdfg(a=a, out=out_ref, M=Mv)
        vsdfg(a=a, out=out_vec, M=Mv)
        assert np.max(np.abs(out_ref - out_vec)) < 1e-12, f"M={Mv}"
