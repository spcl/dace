# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the WCR -> OpenMP ``reduction(...)`` codegen extension.

The CPU codegen emits ``#pragma omp parallel for reduction(op:var)`` for WCR
write-edges that target a true ``Scalar`` descriptor outside a parallel map,
*and* skips the per-edge ``reduce_atomic`` emission for the same target --
the OMP runtime privatizes the variable per thread and tree-reduces at the
end, so an extra atomic add is strictly wasted work.
"""
import os
import shutil

import numpy as np
import pytest

import dace

N = dace.symbol("N")


def _build_wcr_scalar_sum() -> dace.SDFG:
    """Hand-built SDFG: scalar 'acc' accumulated via WCR from a parallel map.

    Done by hand (rather than from a @dace.program) so the accumulator is a
    true Scalar descriptor -- the shape the OMP reduction clause requires.
    """
    sdfg = dace.SDFG("wcr_scalar_sum")
    sdfg.add_array("src", [N], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_scalar("acc", dace.float64, transient=True)

    # init: acc = 0
    init = sdfg.add_state("init", is_start_block=True)
    t0 = init.add_tasklet("seed", {}, {"o"}, "o = 0.0")
    init.add_edge(t0, "o", init.add_write("acc"), None, dace.Memlet("acc[0]"))

    # map: parallel for i: WCR-+= src[i] -> acc
    map_state = sdfg.add_state("map_state")
    sdfg.add_edge(init, map_state, dace.InterstateEdge())
    me, mx = map_state.add_map("m", dict(i="0:N"), schedule=dace.ScheduleType.CPU_Multicore)
    t = map_state.add_tasklet("acc", {"v"}, {"r"}, "r = v")
    src_an = map_state.add_read("src")
    acc_an = map_state.add_write("acc")
    map_state.add_memlet_path(src_an, me, t, dst_conn="v", memlet=dace.Memlet("src[i]"))
    map_state.add_memlet_path(t, mx, acc_an, src_conn="r", memlet=dace.Memlet("acc[0]", wcr="lambda a, b: a + b"))

    # writeback: out[0] = acc
    post = sdfg.add_state("post")
    sdfg.add_edge(map_state, post, dace.InterstateEdge())
    t1 = post.add_tasklet("wb", {"i"}, {"o"}, "o = i")
    post.add_edge(post.add_read("acc"), None, t1, "i", dace.Memlet("acc[0]"))
    post.add_edge(t1, "o", post.add_write("out"), None, dace.Memlet("out[0]"))

    sdfg.validate()
    return sdfg


def _compile_and_read_src(sdfg: dace.SDFG):
    """Compile the SDFG and return ``(csdfg, generated_cpp_source)``."""
    if os.path.exists(sdfg.build_folder):
        shutil.rmtree(sdfg.build_folder)
    csdfg = sdfg.compile()
    src_path = os.path.join(sdfg.build_folder, "src", "cpu", sdfg.name + ".cpp")
    assert os.path.exists(src_path), src_path
    return csdfg, open(src_path).read()


def test_scalar_wcr_emits_omp_reduction_clause():
    """The OMP pragma should include ``reduction(+:acc)`` for the scalar accumulator."""
    sdfg = _build_wcr_scalar_sum()
    csdfg, src = _compile_and_read_src(sdfg)

    # The OMP pragma must include the reduction clause for the scalar accumulator.
    pragma_lines = [l for l in src.splitlines() if "#pragma omp parallel for" in l]
    assert any("reduction(+:" in l for l in pragma_lines), \
        "expected reduction(+:...) clause in the OMP pragma -- got:\n" + "\n".join(pragma_lines)


def test_scalar_wcr_does_not_emit_atomic_for_covered_target():
    """The body must NOT call ``reduce_atomic`` for the WCR target covered by
    the OMP ``reduction(...)`` clause -- the OMP runtime handles privatization +
    final tree-reduce, an extra per-iter atomic is strictly wasted work."""
    sdfg = _build_wcr_scalar_sum()
    csdfg, src = _compile_and_read_src(sdfg)

    # Some ``reduce_atomic`` call may legitimately exist elsewhere in the
    # generated TU (other WCR writes), but specifically NOT on ``acc``.
    bad = [l for l in src.splitlines() if "reduce_atomic" in l and "acc" in l]
    assert not bad, "found per-iter atomic on the OMP-reduction-covered target:\n" + "\n".join(bad)


def test_scalar_wcr_numerically_correct():
    """End-to-end: the OMP-reduction code path computes the sum correctly."""
    sdfg = _build_wcr_scalar_sum()
    csdfg, _ = _compile_and_read_src(sdfg)

    n = 1024
    rng = np.random.default_rng(0)
    src = rng.random(n)
    out = np.array([0.0])
    csdfg(src=src, out=out, N=n)
    assert np.isclose(float(out[0]), float(src.sum()))


def test_multiple_omp_reducible_targets_emit_multiple_clauses():
    """With 2+ scalar WCR targets in one map, each qualifying accumulator gets its own
    ``reduction(op:var)`` clause on the same pragma -- multi-target reduction is valid
    OpenMP (the single-target cap was dropped in c1f38eedc). The atomic path is skipped
    for the covered targets."""
    sdfg = dace.SDFG("wcr_two_scalars")
    sdfg.add_array("src", [N], dace.float64)
    sdfg.add_scalar("acc1", dace.float64, transient=True)
    sdfg.add_scalar("acc2", dace.float64, transient=True)
    sdfg.add_array("out1", [1], dace.float64)
    sdfg.add_array("out2", [1], dace.float64)

    init = sdfg.add_state("init", is_start_block=True)
    init.add_edge(init.add_tasklet("z1", {}, {"o"}, "o = 0.0"), "o", init.add_write("acc1"), None,
                  dace.Memlet("acc1[0]"))
    init.add_edge(init.add_tasklet("z2", {}, {"o"}, "o = 0.0"), "o", init.add_write("acc2"), None,
                  dace.Memlet("acc2[0]"))

    ms = sdfg.add_state("ms")
    sdfg.add_edge(init, ms, dace.InterstateEdge())
    me, mx = ms.add_map("m", dict(i="0:N"), schedule=dace.ScheduleType.CPU_Multicore)
    t = ms.add_tasklet("two", {"v"}, {"r1", "r2"}, "r1 = v; r2 = 2.0 * v")
    ms.add_memlet_path(ms.add_read("src"), me, t, dst_conn="v", memlet=dace.Memlet("src[i]"))
    ms.add_memlet_path(t,
                       mx,
                       ms.add_write("acc1"),
                       src_conn="r1",
                       memlet=dace.Memlet("acc1[0]", wcr="lambda a, b: a + b"))
    ms.add_memlet_path(t,
                       mx,
                       ms.add_write("acc2"),
                       src_conn="r2",
                       memlet=dace.Memlet("acc2[0]", wcr="lambda a, b: a + b"))

    post = sdfg.add_state("post")
    sdfg.add_edge(ms, post, dace.InterstateEdge())
    for tag in ("1", "2"):
        wb = post.add_tasklet("wb" + tag, {"i"}, {"o"}, "o = i")
        post.add_edge(post.add_read("acc" + tag), None, wb, "i", dace.Memlet("acc" + tag + "[0]"))
        post.add_edge(wb, "o", post.add_write("out" + tag), None, dace.Memlet("out" + tag + "[0]"))

    sdfg.validate()
    csdfg, src = _compile_and_read_src(sdfg)
    pragma_lines = [l for l in src.splitlines() if "#pragma omp parallel for" in l]
    assert any("reduction(+:acc1)" in l and "reduction(+:acc2)" in l for l in pragma_lines), \
        "expected reduction(+:acc1) and reduction(+:acc2) on one pragma -- got:\n" + "\n".join(pragma_lines)
    assert "reduce_atomic" not in src, "covered reduction targets must skip the atomic path"


def _build_op_sdfg(op_name: str, wcr: str, dtype: dace.typeclass, init: str) -> dace.SDFG:
    """Scalar accumulator with a configurable WCR op, dtype, and seed value.

    :param op_name: Tag for the SDFG name (no spaces).
    :param wcr: WCR lambda string (e.g. ``'lambda a, b: a + b'``).
    :param dtype: DaCe element type for ``src`` / ``acc`` / ``out``.
    :param init: Tasklet code for the seed write (e.g. ``'o = 0.0'``).
    """
    sdfg = dace.SDFG(f"wcr_scalar_{op_name}")
    sdfg.add_array("src", [N], dtype)
    sdfg.add_array("out", [1], dtype)
    sdfg.add_scalar("acc", dtype, transient=True)

    init_state = sdfg.add_state("init", is_start_block=True)
    t0 = init_state.add_tasklet("seed", {}, {"o"}, init)
    init_state.add_edge(t0, "o", init_state.add_write("acc"), None, dace.Memlet("acc[0]"))

    ms = sdfg.add_state("ms")
    sdfg.add_edge(init_state, ms, dace.InterstateEdge())
    me, mx = ms.add_map("m", dict(i="0:N"), schedule=dace.ScheduleType.CPU_Multicore)
    t = ms.add_tasklet("acc", {"v"}, {"r"}, "r = v")
    ms.add_memlet_path(ms.add_read("src"), me, t, dst_conn="v", memlet=dace.Memlet("src[i]"))
    ms.add_memlet_path(t, mx, ms.add_write("acc"), src_conn="r", memlet=dace.Memlet("acc[0]", wcr=wcr))

    post = sdfg.add_state("post")
    sdfg.add_edge(ms, post, dace.InterstateEdge())
    wb = post.add_tasklet("wb", {"i"}, {"o"}, "o = i")
    post.add_edge(post.add_read("acc"), None, wb, "i", dace.Memlet("acc[0]"))
    post.add_edge(wb, "o", post.add_write("out"), None, dace.Memlet("out[0]"))
    sdfg.validate()
    return sdfg


_FLOAT_RNG = np.random.default_rng(42)


def _f64_rand(n):
    """Bounded floats so Product / Min / Max don't overflow or saturate."""
    return _FLOAT_RNG.uniform(0.9, 1.1, size=n)


def _i32_rand(n):
    return np.random.default_rng(7).integers(0, 16, size=n, dtype=np.int32)


def _bool_rand_mostly_true(n):
    """Mostly True so Logical_And does not degenerate to a constant False oracle."""
    arr = np.random.default_rng(11).choice([True, False], size=n, p=[0.95, 0.05])
    return arr.astype(np.bool_)


def _bool_rand_mostly_false(n):
    """Mostly False so Logical_Or does not degenerate to a constant True oracle."""
    arr = np.random.default_rng(13).choice([True, False], size=n, p=[0.05, 0.95])
    return arr.astype(np.bool_)


# Tuple layout: op_name, wcr_lambda, expected_clause_op, dtype, init_tasklet_body,
# np_input_generator, np_oracle_callable, scalar_compare_callable. The
# expected_clause_op is the EXACT substring that must appear in the
# ``reduction(<op>:acc)`` clause of the emitted OMP pragma, jointly pinning
# detect_reduction_type's WCR-lambda -> ReductionType mapping AND the
# _REDUCTION_TO_OMP_OP table.
PER_OP_CASES = [
    ("sum", "lambda a, b: a + b", "+", dace.float64, "o = 0.0", _f64_rand, lambda x: float(x.sum()), np.isclose),
    ("product", "lambda a, b: a * b", "*", dace.float64, "o = 1.0", _f64_rand, lambda x: float(x.prod()), np.isclose),
    ("min", "lambda a, b: min(a, b)", "min", dace.float64, "o = 1e9", _f64_rand, lambda x: float(x.min()), np.isclose),
    ("max", "lambda a, b: max(a, b)", "max", dace.float64, "o = -1e9", _f64_rand, lambda x: float(x.max()), np.isclose),
    ("band", "lambda a, b: a & b", "&", dace.int32, "o = -1", _i32_rand, lambda x: int(np.bitwise_and.reduce(x)),
     lambda a, b: int(a) == int(b)),
    ("bor", "lambda a, b: a | b", "|", dace.int32, "o = 0", _i32_rand, lambda x: int(np.bitwise_or.reduce(x)),
     lambda a, b: int(a) == int(b)),
    ("bxor", "lambda a, b: a ^ b", "^", dace.int32, "o = 0", _i32_rand, lambda x: int(np.bitwise_xor.reduce(x)),
     lambda a, b: int(a) == int(b)),
    ("land", "lambda a, b: a and b", "&&", dace.int32, "o = 1", lambda n: _bool_rand_mostly_true(n).astype(np.int32),
     lambda x: int(bool(np.all(x))), lambda a, b: bool(int(a)) == bool(int(b))),
    ("lor", "lambda a, b: a or b", "||", dace.int32, "o = 0", lambda n: _bool_rand_mostly_false(n).astype(np.int32),
     lambda x: int(bool(np.any(x))), lambda a, b: bool(int(a)) == bool(int(b))),
]


@pytest.mark.parametrize("op_name,wcr,expected_op,dtype,init,gen_input,oracle,compare",
                         PER_OP_CASES,
                         ids=[c[0] for c in PER_OP_CASES])
def test_per_operator_emits_correct_omp_reduction_clause(op_name, wcr, expected_op, dtype, init, gen_input, oracle,
                                                         compare):
    """Each supported WCR op must produce the matching ``reduction(<op>:acc)``
    clause AND compute the right numerical result. Pins three things at once:

    1. ``detect_reduction_type`` correctly classifies the WCR lambda.
    2. ``_REDUCTION_TO_OMP_OP`` maps that classification to the right OpenMP
       operator token.
    3. End-to-end runtime: the OMP per-thread privatize + final tree-reduce
       computes the same result as the un-parallelised oracle.
    """
    sdfg = _build_op_sdfg(op_name, wcr, dtype, init)
    csdfg, src = _compile_and_read_src(sdfg)

    pragma_lines = [l for l in src.splitlines() if "#pragma omp parallel for" in l]
    target_clause = f"reduction({expected_op}:"
    assert any(
        target_clause in l
        for l in pragma_lines), (f"expected '{target_clause}' clause for op '{op_name}' (wcr={wcr!r}); got pragmas:\n" +
                                 "\n".join(pragma_lines))

    n = 1024
    src_arr = gen_input(n)
    out = np.zeros(1, dtype=src_arr.dtype)
    csdfg(src=src_arr, out=out, N=n)
    expected = oracle(src_arr)
    assert compare(out[0], expected), f"{op_name}: got {out[0]}, expected {expected}"


@pytest.mark.parametrize("op_name,wcr,expected_op,dtype,init,gen_input,oracle,compare",
                         PER_OP_CASES,
                         ids=[c[0] for c in PER_OP_CASES])
def test_per_operator_suppresses_atomic_on_covered_target(op_name, wcr, expected_op, dtype, init, gen_input, oracle,
                                                          compare):
    """For each supported op, the per-edge ``reduce_atomic`` must be skipped
    on the OMP-reduction-covered target: the runtime's per-thread copy + final
    tree-reduce makes an extra atomic on top strictly wasted work (and would
    be wrong if the implementation privatised by value rather than pointer).
    """
    sdfg = _build_op_sdfg(op_name, wcr, dtype, init)
    _, src = _compile_and_read_src(sdfg)
    bad = [l for l in src.splitlines() if "reduce_atomic" in l and "acc" in l]
    assert not bad, f"{op_name}: per-iter atomic on covered target:\n" + "\n".join(bad)


def test_unsupported_op_falls_back_to_atomic():
    """A WCR op outside _REDUCTION_TO_OMP_OP (here subtraction, which OpenMP
    does NOT support as a reduction operator -- ``a - b`` is not associative)
    must fall through to the existing atomic-add path. No reduction clause;
    the atomic is the only correctness guarantee."""
    sdfg = _build_op_sdfg("sub", "lambda a, b: a - b", dace.float64, "o = 0.0")
    _, src = _compile_and_read_src(sdfg)
    pragma_lines = [l for l in src.splitlines() if "#pragma omp parallel for" in l]
    assert not any("reduction(" in l
                   for l in pragma_lines), ("expected NO reduction clause for an unsupported op; got:\n" +
                                            "\n".join(pragma_lines))
    assert any("reduce_atomic" in l and "acc" in l for l in src.splitlines()), \
        "expected reduce_atomic on 'acc' as fallback when no reduction clause is emitted"


def test_length_one_array_target_falls_back_to_atomic():
    """A length-1 ``Array`` accumulator (``arr[0]``) is emitted by DaCe as a
    ``T*`` pointer slot, which OpenMP's ``reduction(...)`` clause cannot use
    (it needs a scalar VARIABLE). Detection must refuse this case; the atomic
    path must fire instead. ``PrivatizeReductionAccumulator`` is the upstream
    rewrite that converts such targets into a transient ``Scalar``; without
    it, atomic is the correct fallback."""
    sdfg = dace.SDFG("wcr_len1_array_target")
    sdfg.add_array("src", [N], dace.float64)
    sdfg.add_array("acc", [1], dace.float64, transient=True)  # length-1 Array, NOT Scalar
    sdfg.add_array("out", [1], dace.float64)

    init = sdfg.add_state("init", is_start_block=True)
    t0 = init.add_tasklet("seed", {}, {"o"}, "o = 0.0")
    init.add_edge(t0, "o", init.add_write("acc"), None, dace.Memlet("acc[0]"))

    ms = sdfg.add_state("ms")
    sdfg.add_edge(init, ms, dace.InterstateEdge())
    me, mx = ms.add_map("m", dict(i="0:N"), schedule=dace.ScheduleType.CPU_Multicore)
    t = ms.add_tasklet("acc", {"v"}, {"r"}, "r = v")
    ms.add_memlet_path(ms.add_read("src"), me, t, dst_conn="v", memlet=dace.Memlet("src[i]"))
    ms.add_memlet_path(t, mx, ms.add_write("acc"), src_conn="r", memlet=dace.Memlet("acc[0]", wcr="lambda a, b: a + b"))

    post = sdfg.add_state("post")
    sdfg.add_edge(ms, post, dace.InterstateEdge())
    wb = post.add_tasklet("wb", {"i"}, {"o"}, "o = i")
    post.add_edge(post.add_read("acc"), None, wb, "i", dace.Memlet("acc[0]"))
    post.add_edge(wb, "o", post.add_write("out"), None, dace.Memlet("out[0]"))
    sdfg.validate()

    _, src = _compile_and_read_src(sdfg)
    pragma_lines = [l for l in src.splitlines() if "#pragma omp parallel for" in l]
    assert not any("reduction(" in l
                   for l in pragma_lines), ("expected NO reduction clause for length-1 Array target; got:\n" +
                                            "\n".join(pragma_lines))


def test_persistent_scalar_target_falls_back_to_atomic():
    """A WCR scalar accumulator that is PERSISTENT is emitted as a state-struct
    member (``__state->acc``), which is not a valid ``reduction(op:var)`` lvalue.
    Detection must refuse it (no reduction clause) and the atomic path is the
    correct fallback. Regression: canonicalize's finalize made the go_fast
    ``trace`` accumulator persistent, so codegen emitted
    ``reduction(+:__state->__0_trace)`` and the kernel failed to compile."""
    sdfg = _build_wcr_scalar_sum()
    sdfg.arrays["acc"].lifetime = dace.dtypes.AllocationLifetime.Persistent
    sdfg.arrays["acc"].storage = dace.dtypes.StorageType.CPU_Heap
    sdfg.validate()

    csdfg, src = _compile_and_read_src(sdfg)
    pragma_lines = [l for l in src.splitlines() if "#pragma omp parallel for" in l]
    assert not any("reduction(" in l for l in pragma_lines), \
        ("expected NO reduction clause for a persistent (state-resident) target; got:\n" + "\n".join(pragma_lines))
    # The illegal ``reduction(+:__state->...)`` lvalue must never be emitted.
    assert "reduction(+:__state->" not in src, "emitted an illegal OMP reduction on a state-struct member"
    # The atomic path is the fallback that keeps the persistent reduction correct.
    assert any("reduce_atomic" in l for l in src.splitlines()), \
        "expected reduce_atomic fallback for the persistent target"

    # End-to-end: the atomic fallback still computes the sum correctly.
    n = 1024
    rng = np.random.default_rng(0)
    src_arr = rng.random(n)
    out = np.array([0.0])
    csdfg(src=src_arr, out=out, N=n)
    assert np.isclose(float(out[0]), float(src_arr.sum()))


def test_mixed_copy_and_reduce_map():
    """A single parallel map that BOTH copies element-wise (``a[i] = b[i]``) AND
    reduces (``d += c[i]``, a scalar WCR) must codegen correctly: the OMP pragma gets
    a ``reduction(+:d)`` clause for the scalar accumulator while the parallel-for still
    writes ``a[i]`` element-wise. This is the azimint-shaped "copy + reduce in one map"
    that the canon round-trip mishandles -- here we pin that the *codegen* of the
    well-formed (WCR) shape is correct."""
    N = dace.symbol("N")
    sdfg = dace.SDFG("mixed_copy_reduce")
    for arr in ("b", "c", "a"):
        sdfg.add_array(arr, [N], dace.float64)
    sdfg.add_scalar("d", dace.float64, transient=True)
    sdfg.add_array("out", [1], dace.float64)

    init = sdfg.add_state("init", is_start_block=True)
    t0 = init.add_tasklet("seed", {}, {"o"}, "o = 0.0")
    init.add_edge(t0, "o", init.add_write("d"), None, dace.Memlet("d[0]"))

    ms = sdfg.add_state("ms")
    sdfg.add_edge(init, ms, dace.InterstateEdge())
    me, mx = ms.add_map("m", dict(i="0:N"), schedule=dace.ScheduleType.CPU_Multicore)
    # element-wise copy a[i] = b[i]
    tcopy = ms.add_tasklet("copy", {"bi"}, {"ai"}, "ai = bi")
    ms.add_memlet_path(ms.add_read("b"), me, tcopy, dst_conn="bi", memlet=dace.Memlet("b[i]"))
    ms.add_memlet_path(tcopy, mx, ms.add_write("a"), src_conn="ai", memlet=dace.Memlet("a[i]"))
    # scalar reduction d += c[i]
    tred = ms.add_tasklet("red", {"ci"}, {"r"}, "r = ci")
    ms.add_memlet_path(ms.add_read("c"), me, tred, dst_conn="ci", memlet=dace.Memlet("c[i]"))
    ms.add_memlet_path(tred, mx, ms.add_write("d"), src_conn="r", memlet=dace.Memlet("d[0]", wcr="lambda a, b: a + b"))

    post = sdfg.add_state("post")
    sdfg.add_edge(ms, post, dace.InterstateEdge())
    twb = post.add_tasklet("wb", {"i"}, {"o"}, "o = i")
    post.add_edge(post.add_read("d"), None, twb, "i", dace.Memlet("d[0]"))
    post.add_edge(twb, "o", post.add_write("out"), None, dace.Memlet("out[0]"))
    sdfg.validate()

    csdfg, src = _compile_and_read_src(sdfg)
    pragma_lines = [l for l in src.splitlines() if "#pragma omp parallel for" in l]
    assert any("reduction(+:" in l for l in pragma_lines), \
        ("expected reduction(+:d) on the copy+reduce map; got:\n" + "\n".join(pragma_lines))

    n = 256
    rng = np.random.default_rng(0)
    b, c = rng.random(n), rng.random(n)
    a, out = np.zeros(n), np.zeros(1)
    csdfg(b=b, c=c, a=a, out=out, N=n)
    assert np.allclose(a, b), "element-wise copy a[i]=b[i] wrong"
    assert np.isclose(float(out[0]), float(c.sum())), "scalar reduction d=sum(c) wrong"


def test_mixed_copy_and_product_reduce_detects_star_op():
    """Op detection from the WCR edge: a map that copies (``a[i] = b[i]``) AND multiplies
    (``p *= c[i]``) must emit ``reduction(*:p)`` -- the operator is read from the WCR
    lambda (``a * b``), NOT hardcoded to ``+``. The element-wise copy coexists."""
    N = dace.symbol("N")
    sdfg = dace.SDFG("mixed_copy_prod")
    for arr in ("b", "c", "a"):
        sdfg.add_array(arr, [N], dace.float64)
    sdfg.add_scalar("p", dace.float64, transient=True)
    sdfg.add_array("out", [1], dace.float64)

    init = sdfg.add_state("init", is_start_block=True)
    t0 = init.add_tasklet("seed", {}, {"o"}, "o = 1.0")  # product identity
    init.add_edge(t0, "o", init.add_write("p"), None, dace.Memlet("p[0]"))

    ms = sdfg.add_state("ms")
    sdfg.add_edge(init, ms, dace.InterstateEdge())
    me, mx = ms.add_map("m", dict(i="0:N"), schedule=dace.ScheduleType.CPU_Multicore)
    tcopy = ms.add_tasklet("copy", {"bi"}, {"ai"}, "ai = bi")
    ms.add_memlet_path(ms.add_read("b"), me, tcopy, dst_conn="bi", memlet=dace.Memlet("b[i]"))
    ms.add_memlet_path(tcopy, mx, ms.add_write("a"), src_conn="ai", memlet=dace.Memlet("a[i]"))
    tred = ms.add_tasklet("red", {"ci"}, {"r"}, "r = ci")
    ms.add_memlet_path(ms.add_read("c"), me, tred, dst_conn="ci", memlet=dace.Memlet("c[i]"))
    ms.add_memlet_path(tred, mx, ms.add_write("p"), src_conn="r", memlet=dace.Memlet("p[0]", wcr="lambda a, b: a * b"))

    post = sdfg.add_state("post")
    sdfg.add_edge(ms, post, dace.InterstateEdge())
    twb = post.add_tasklet("wb", {"i"}, {"o"}, "o = i")
    post.add_edge(post.add_read("p"), None, twb, "i", dace.Memlet("p[0]"))
    post.add_edge(twb, "o", post.add_write("out"), None, dace.Memlet("out[0]"))
    sdfg.validate()

    csdfg, src = _compile_and_read_src(sdfg)
    pragma_lines = [l for l in src.splitlines() if "#pragma omp parallel for" in l]
    assert any("reduction(*:" in l for l in pragma_lines), \
        ("expected reduction(*:p) (op detected from WCR lambda ``a*b``); got:\n" + "\n".join(pragma_lines))
    assert not any("reduction(+:" in l for l in pragma_lines), "must not emit '+'; the WCR op is '*'"

    n = 64
    rng = np.random.default_rng(1)
    b = rng.random(n)
    c = rng.random(n) * 0.5 + 0.75  # keep the product well-conditioned
    a, out = np.zeros(n), np.zeros(1)
    csdfg(b=b, c=c, a=a, out=out, N=n)
    assert np.allclose(a, b), "element-wise copy a[i]=b[i] wrong"
    assert np.isclose(float(out[0]), float(np.prod(c))), "product reduction p=prod(c) wrong"


# ---------------------------------------------------------------------------
# OpenMP array-section reduction  reduction(op:A[0:n])  (openmp_array_reductions flag)
# ---------------------------------------------------------------------------
KK, NR, NM = 200, 5, 7


def _build_wcr_array_sum(dtype, flag: bool, wcr: str = "lambda a, b: a + b", strides=None) -> dace.SDFG:
    """Hand-built target shape: an outer CPU_Multicore reduction map whose body writes
    a WHOLE array ``A`` with WCR -- ``outer(k){ inner(i,j){ A[i,j] op= X[k,i,j] } }``.

    The WCR memlet at the OUTER map exit covers ``A[0:NR,0:NM]`` (independent of ``k``),
    exactly the shape ``_collect_omp_reductions`` turns into ``reduction(op:A[0:NR*NM])``
    when ``openmp_array_reductions`` is on. ``strides`` forces a non-contiguous layout
    (to exercise the atomic fallback).
    """
    sd = dace.SDFG("wcr_arr_%s_%d_%s" % (dtype.to_string(), int(flag), "s" if strides else "c"))
    sd.add_array("X", [KK, NR, NM], dtype)
    if strides is None:
        sd.add_array("A", [NR, NM], dtype)
    else:
        sd.add_array("A", [NR, NM], dtype, strides=strides)
    st = sd.add_state()
    oe, ox = st.add_map("outer", dict(k="0:%d" % KK), schedule=dace.ScheduleType.CPU_Multicore)
    ie, ix = st.add_map("inner", dict(i="0:%d" % NR, j="0:%d" % NM))
    t = st.add_tasklet("acc", {"xin"}, {"aout"}, "aout = xin")
    st.add_memlet_path(st.add_read("X"), oe, ie, t, dst_conn="xin", memlet=dace.Memlet(data="X", subset="k, i, j"))
    st.add_memlet_path(t,
                       ix,
                       ox,
                       st.add_write("A"),
                       src_conn="aout",
                       memlet=dace.Memlet(data="A", subset="i, j", wcr=wcr))
    sd.validate()
    sd.openmp_array_reductions = flag
    return sd


def _run_arr(sd, X, A0):
    A = A0.copy()
    sd(X=X, A=A)
    return A


def test_array_wcr_emits_omp_array_section_clause():
    """Flag on: a whole-buffer WCR accumulator of a parallel map becomes
    ``reduction(+:A[0:35])`` and the covered inner write skips ``reduce_atomic``."""
    sd = _build_wcr_array_sum(dace.float64, flag=True)
    _, src = _compile_and_read_src(sd)
    pragma = [l for l in src.splitlines() if "#pragma omp parallel for" in l]
    assert any("reduction(+:A[0:" in l for l in pragma), \
        "expected reduction(+:A[0:n]) array-section clause -- got:\n" + "\n".join(pragma)
    assert not any("reduce_atomic" in l and "A " in l for l in src.splitlines()), \
        "covered array target must not use the atomic path"


def test_array_wcr_flag_off_falls_back_to_atomic():
    """Flag off (default): no array-section clause; the atomic path emits as before."""
    sd = _build_wcr_array_sum(dace.float64, flag=False)
    _, src = _compile_and_read_src(sd)
    pragma = [l for l in src.splitlines() if "#pragma omp parallel for" in l]
    assert not any("[0:" in l and "reduction(" in l for l in pragma), \
        "flag off must NOT emit an array-section reduction clause -- got:\n" + "\n".join(pragma)
    assert "reduce_atomic" in src, "flag off must keep the atomic path"


def test_array_wcr_numerically_correct_omp4():
    """Flag on, OMP=4: the array reduction is race-free and bit-exact vs the sequential sum,
    and matches the flag-off atomic result."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((KK, NR, NM))
    A0 = rng.standard_normal((NR, NM))
    ref = A0 + X.sum(axis=0)
    old = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "4"
    try:
        got_on = _run_arr(_build_wcr_array_sum(dace.float64, flag=True), X, A0)
        got_off = _run_arr(_build_wcr_array_sum(dace.float64, flag=False), X, A0)
    finally:
        if old is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old
    assert np.allclose(got_on, ref), "flag-on array reduction numerically wrong"
    assert np.allclose(got_off, ref), "flag-off atomic reduction numerically wrong"


def test_complex_array_wcr_emits_declare_reduction_and_is_correct():
    """Complex element type: flag on emits a ``#pragma omp declare reduction`` for
    ``dace::complex128`` (OpenMP has no built-in complex reduction) plus the array-section
    clause, and is bit-exact at OMP=4."""
    sd = _build_wcr_array_sum(dace.complex128, flag=True)
    _, src = _compile_and_read_src(sd)
    assert "declare reduction(+ : dace::complex128" in src, \
        "expected complex declare-reduction directive"
    assert any("reduction(+:A[0:" in l for l in src.splitlines() if "#pragma omp parallel for" in l)

    rng = np.random.default_rng(3)
    X = (rng.standard_normal((KK, NR, NM)) + 1j * rng.standard_normal((KK, NR, NM)))
    A0 = np.zeros((NR, NM), np.complex128)
    ref = A0 + X.sum(axis=0)
    old = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "4"
    try:
        got = _run_arr(_build_wcr_array_sum(dace.complex128, flag=True), X, A0)
    finally:
        if old is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old
    assert np.allclose(got, ref), "complex array reduction numerically wrong"


def test_noncontiguous_array_target_falls_back_to_atomic():
    """Safety: a non-contiguous (transposed-stride) buffer can't be reduced as a flat
    ``A[0:n]`` section, so even with the flag on it falls back to the atomic path."""
    sd = _build_wcr_array_sum(dace.float64, flag=True, strides=[1, NR])
    _, src = _compile_and_read_src(sd)
    pragma = [l for l in src.splitlines() if "#pragma omp parallel for" in l]
    assert not any("[0:" in l and "reduction(" in l for l in pragma), \
        "non-contiguous buffer must NOT take the array-section clause -- got:\n" + "\n".join(pragma)


def test_complex_product_declare_uses_identity_one():
    """A complex ``*`` reduction declares identity 1 (not 0)."""
    sd = _build_wcr_array_sum(dace.complex128, flag=True, wcr="lambda a, b: a * b")
    _, src = _compile_and_read_src(sd)
    assert "declare reduction(* : dace::complex128" in src
    assert "initializer(omp_priv = dace::complex128(1))" in src


if __name__ == "__main__":
    test_mixed_copy_and_reduce_map()
    test_mixed_copy_and_product_reduce_detects_star_op()
    test_persistent_scalar_target_falls_back_to_atomic()
    test_scalar_wcr_emits_omp_reduction_clause()
    test_scalar_wcr_does_not_emit_atomic_for_covered_target()
    test_scalar_wcr_numerically_correct()
    test_multiple_omp_reducible_targets_emit_multiple_clauses()
    for case in PER_OP_CASES:
        test_per_operator_emits_correct_omp_reduction_clause(*case)
        test_per_operator_suppresses_atomic_on_covered_target(*case)
    test_unsupported_op_falls_back_to_atomic()
    test_length_one_array_target_falls_back_to_atomic()
    test_array_wcr_emits_omp_array_section_clause()
    test_array_wcr_flag_off_falls_back_to_atomic()
    test_array_wcr_numerically_correct_omp4()
    test_complex_array_wcr_emits_declare_reduction_and_is_correct()
    test_noncontiguous_array_target_falls_back_to_atomic()
    test_complex_product_declare_uses_identity_one()
