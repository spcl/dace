# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``gather_intrinsic`` / ``scatter_intrinsic`` ``VectorizeCPU`` knobs.

Three concerns:

* ``test_gather_intrinsic_emission`` — on a clean ``a[i] = b[idx[i]]``
  gather the knob must actually change emitted code: the unmasked
  main-loop ``gather<`` intrinsic appears iff ``gather_intrinsic`` is
  ``True``; the masked remainder ``gather_masked<`` is forced whenever
  ``remainder_strategy == "masked"`` (a per-lane scalar fan would fault
  on inactive lanes), independent of the knob. Also numerically checked.
* ``test_gather_scatter_numeric`` — the knob never changes results, on a
  broader kernel set crossed with both remainder strategies and a
  vector-width-divisible (64) and non-divisible (65) length.
* ``test_scatter_collapse_is_a_known_gap`` — documents that
  ``DetectScatter`` does not currently collapse an ``a[ip[i]]`` scatter
  fan in this pipeline, so ``scatter_intrinsic`` is wired but inert for
  these patterns (numerics stay correct via the scalar fan). The assert
  flips when scatter collapse is implemented.

Index arrays are the identity ``arange`` so scatters are a valid
permutation and every access is in bounds.
"""
import copy

import numpy as np
import pytest

import dace
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

LEN_1D = dace.symbol("LEN_1D")


@dace.program
def vag(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = b[ip[i]]


@dace.program
def idx_table_gather(out: dace.float64[LEN_1D], table: dace.float64[LEN_1D], idx: dace.int32[LEN_1D]):
    # Non-TSVC: a plain indirect-index gather from a separate table.
    for i in range(LEN_1D):
        out[i] = table[idx[i]] * 2.0 + 1.0


@dace.program
def s491(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
         ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        a[ip[i]] = b[i] + c[i] * d[i]


@dace.program
def s4113(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        a[ip[i]] = b[ip[i]] + c[i]


@dace.program
def s4115(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D], sum_out: dace.float64[1]):
    sum_val = 0.0
    for i in range(LEN_1D):
        sum_val = sum_val + a[i] * b[ip[i]]
    sum_out[0] = sum_val


_GATHER_KERNELS = {
    "vag": (vag, ("a", "b", "ip")),
    "idx_table_gather": (idx_table_gather, ("out", "table", "idx")),
}
_NUMERIC_KERNELS = {
    **_GATHER_KERNELS,
    "s491": (s491, ("a", "b", "c", "d", "ip")),
    "s4113": (s4113, ("a", "b", "c", "ip")),
    "s4115": (s4115, ("a", "b", "ip", "sum_out")),
}


def _alloc(name, n, rng):
    if name in ("ip", "idx"):
        return np.arange(n, dtype=np.int32)
    if name == "sum_out":
        return np.zeros(1, dtype=np.float64)
    return rng.random(n).astype(np.float64)


def _vectorize(prog,
               argnames,
               len_1d,
               *,
               gather_intrinsic=True,
               scatter_intrinsic=True,
               remainder_strategy="scalar",
               name_prefix=""):
    """Build ref + vectorized SDFGs and return ``(ref_sdfg, vec_sdfg, ref_args, vec_args)``.

    :param name_prefix: disambiguates the generated SDFG (hence the
        ``.dacecache`` build dir) so concurrent ``-n`` workers running
        different test functions on the same kernel/params do not race
        on a shared build directory.
    """
    rng = np.random.default_rng(seed=len_1d)
    ref_args = {nm: _alloc(nm, len_1d, rng) for nm in argnames}
    vec_args = {nm: arr.copy() for nm, arr in ref_args.items()}
    tag = f"{name_prefix}{prog.name}_g{int(gather_intrinsic)}_s{int(scatter_intrinsic)}_{remainder_strategy}_{len_1d}"
    sdfg = copy.deepcopy(prog.to_sdfg(simplify=False))
    sdfg.name = f"ref_{tag}"
    sdfg.simplify()
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()
    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"vec_{tag}"
    VectorizeCPU(vector_width=8,
                 fail_on_unvectorizable=False,
                 remainder_strategy=remainder_strategy,
                 use_fp_factor=False,
                 branch_normalization=True,
                 gather_intrinsic=gather_intrinsic,
                 scatter_intrinsic=scatter_intrinsic).apply_pass(vsdfg, {})
    return sdfg, vsdfg, ref_args, vec_args


def _tasklet_code(sdfg):
    return "\n".join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))


@pytest.mark.parametrize("kernel_name", list(_GATHER_KERNELS))
@pytest.mark.parametrize("gather_intrinsic", [True, False])
@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("len_1d", [64, 65])
def test_gather_intrinsic_emission(kernel_name, gather_intrinsic, remainder_strategy, len_1d):
    prog, argnames = _GATHER_KERNELS[kernel_name]
    sdfg, vsdfg, ref_args, vec_args = _vectorize(prog,
                                                 argnames,
                                                 len_1d,
                                                 gather_intrinsic=gather_intrinsic,
                                                 remainder_strategy=remainder_strategy,
                                                 name_prefix="emit_")
    code = _tasklet_code(vsdfg)
    has_unmasked = "gather<" in code.replace("gather_masked<", "")
    has_masked = "gather_masked<" in code

    # The main loop uses the gather intrinsic iff the knob is on.
    assert has_unmasked is gather_intrinsic, \
        f"gather<: expected {gather_intrinsic}, got {has_unmasked} ({kernel_name} {remainder_strategy} {len_1d})"
    # The masked vector remainder always collapses to the intrinsic
    # (a per-lane scalar fan would fault on inactive lanes), regardless
    # of the knob.
    assert has_masked is (remainder_strategy == "masked"), \
        f"gather_masked<: expected {remainder_strategy == 'masked'}, got {has_masked}"

    sdfg.compile()(**ref_args, LEN_1D=len_1d)
    vsdfg.compile()(**vec_args, LEN_1D=len_1d)
    for nm in argnames:
        diff = np.max(np.abs(ref_args[nm] - vec_args[nm]))
        assert diff < 1e-10, f"{kernel_name}/{nm}: max abs diff = {diff}"


@pytest.mark.parametrize("kernel_name", list(_NUMERIC_KERNELS))
@pytest.mark.parametrize("gather_intrinsic", [True, False])
@pytest.mark.parametrize("scatter_intrinsic", [True, False])
@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("len_1d", [64, 65])
def test_gather_scatter_numeric(kernel_name, gather_intrinsic, scatter_intrinsic, remainder_strategy, len_1d):
    prog, argnames = _NUMERIC_KERNELS[kernel_name]
    try:
        sdfg, vsdfg, ref_args, vec_args = _vectorize(prog,
                                                     argnames,
                                                     len_1d,
                                                     gather_intrinsic=gather_intrinsic,
                                                     scatter_intrinsic=scatter_intrinsic,
                                                     remainder_strategy=remainder_strategy,
                                                     name_prefix="num_")
    except NotImplementedError as ex:
        pytest.skip(f"vectorize NotImplementedError on {kernel_name}: {ex}")
    sdfg.compile()(**ref_args, LEN_1D=len_1d)
    vsdfg.compile()(**vec_args, LEN_1D=len_1d)
    for nm in argnames:
        diff = np.max(np.abs(ref_args[nm] - vec_args[nm]))
        assert diff < 1e-10, f"{kernel_name}/{nm}: max abs diff = {diff}"


@pytest.mark.parametrize("kernel_name", ["s491", "s4113"])
def test_scatter_collapse_is_a_known_gap(kernel_name):
    # DetectScatter does not currently match an ``a[ip[i]]`` scatter fan,
    # so the scatter intrinsic is never emitted for these even with
    # scatter_intrinsic=True. The knob is wired and numerically safe (the
    # scalar fan is correct); this asserts the *current* gap so it flips
    # loudly when scatter collapse is implemented.
    prog, argnames = _NUMERIC_KERNELS[kernel_name]
    _, vsdfg, _, _ = _vectorize(prog,
                                argnames,
                                64,
                                scatter_intrinsic=True,
                                remainder_strategy="masked",
                                name_prefix="gap_")
    code = _tasklet_code(vsdfg)
    assert "scatter<" not in code.replace("scatter_masked<", "") and "scatter_masked<" not in code, \
        f"scatter collapse now fires for {kernel_name} — update test_gather_intrinsic_emission to cover scatter"
