# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``gather_intrinsic`` / ``scatter_intrinsic`` ``VectorizeCPU`` knobs.

The gather and scatter intrinsic paths are symmetric: a ``_packed``
buffer with a per-lane ``assign_<i>`` fan that ``DetectGather`` /
``DetectScatter`` collapse to the ``gather`` / ``scatter`` intrinsic.
The knob controls the *main* loop (``True`` -> intrinsic, ``False`` ->
per-lane scalar fan); the masked vector remainder always collapses to
the intrinsic regardless (a scalar fan would fault on inactive lanes).

An indirect *write* ``a[ip[i]]`` is only conflict-free when ``ip`` has no
duplicate indices, so plain ``LoopToMap`` conservatively refuses it; the
test opts in with ``permissive=True`` (the same permutation contract the
indirect *read* gather path already relies on, and the index arrays here
are the identity ``arange``, a valid permutation with every access in
bounds).

* ``test_gather_intrinsic_emission`` / ``test_scatter_intrinsic_emission``
  — the knob must change emitted code: the unmasked main-loop ``op<``
  intrinsic appears iff the knob is ``True``; the masked-remainder
  ``op_masked<`` is forced iff ``remainder_strategy == "masked"``. Also
  numerically checked against a non-transformed reference.
* ``test_gather_scatter_numeric`` — the knobs never change results, over a
  broader kernel set crossed with both remainder strategies and a
  vector-width-divisible (64) and non-divisible (65) length.
"""
import copy

import numpy as np
import pytest

import dace
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

LEN_1D = dace.symbol("LEN_1D")

# vag / vas / s4113 / s491 / s4115 are TSVC kernels — import the canonical defs
# from the corpus rather than re-declaring identical bodies (dedup; prefer tsvc).
from tests.passes.vectorization.tsvc_1d.test_misc import (  # noqa: E402
    vag_d_single as vag,
    vas_d_single as vas,
    s4113_d_single as s4113,
)
from tests.passes.vectorization.tsvc_1d.test_selected import (  # noqa: E402
    dace_s491 as s491,
    dace_s4115_inner as s4115,
)


@dace.program
def idx_table_gather(out: dace.float64[LEN_1D], table: dace.float64[LEN_1D], idx: dace.int32[LEN_1D]):
    # Non-TSVC: a plain indirect-index gather from a separate table.
    for i in range(LEN_1D):
        out[i] = table[idx[i]] * 2.0 + 1.0


_GATHER_KERNELS = {
    "vag": (vag, ("a", "b", "ip")),
    "idx_table_gather": (idx_table_gather, ("out", "table", "idx")),
}
_SCATTER_KERNELS = {
    "vas": (vas, ("a", "b", "ip")),
    "s491": (s491, ("a", "b", "c", "d", "ip")),
}
_NUMERIC_KERNELS = {
    **_GATHER_KERNELS,
    **_SCATTER_KERNELS,
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
    """Build ref + vectorized SDFGs; return ``(ref_sdfg, vec_sdfg, ref_args, vec_args)``.

    ``LoopToMap`` is applied ``permissive=True`` so an indirect-write
    scatter loop maps (and flows through the same packed-fan path as
    gather); the identity index arrays make that conflict-free.

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
    sdfg.apply_transformations_repeated(LoopToMap, permissive=True)
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


def _assert_emission(vsdfg, *, op, knob_on, masked_remainder):
    code = _tasklet_code(vsdfg)
    has_unmasked = f"{op}<" in code.replace(f"{op}_masked<", "")
    has_masked = f"{op}_masked<" in code
    assert has_unmasked is knob_on, \
        f"{op}<: expected unmasked-present={knob_on}, got {has_unmasked}"
    assert has_masked is masked_remainder, \
        f"{op}_masked<: expected masked-present={masked_remainder}, got {has_masked}"


def _check_numeric(sdfg, vsdfg, ref_args, vec_args, argnames, len_1d, tag):
    sdfg.compile()(**ref_args, LEN_1D=len_1d)
    vsdfg.compile()(**vec_args, LEN_1D=len_1d)
    for nm in argnames:
        diff = np.max(np.abs(ref_args[nm] - vec_args[nm]))
        assert diff < 1e-10, f"{tag}/{nm}: max abs diff = {diff}"


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
                                                 name_prefix="gemit_")
    _assert_emission(vsdfg, op="gather", knob_on=gather_intrinsic, masked_remainder=(remainder_strategy == "masked"))
    _check_numeric(sdfg, vsdfg, ref_args, vec_args, argnames, len_1d, kernel_name)


@pytest.mark.parametrize("kernel_name", list(_SCATTER_KERNELS))
@pytest.mark.parametrize("scatter_intrinsic", [True, False])
@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("len_1d", [64, 65])
def test_scatter_intrinsic_emission(kernel_name, scatter_intrinsic, remainder_strategy, len_1d):
    prog, argnames = _SCATTER_KERNELS[kernel_name]
    sdfg, vsdfg, ref_args, vec_args = _vectorize(prog,
                                                 argnames,
                                                 len_1d,
                                                 scatter_intrinsic=scatter_intrinsic,
                                                 remainder_strategy=remainder_strategy,
                                                 name_prefix="semit_")
    _assert_emission(vsdfg, op="scatter", knob_on=scatter_intrinsic, masked_remainder=(remainder_strategy == "masked"))
    _check_numeric(sdfg, vsdfg, ref_args, vec_args, argnames, len_1d, kernel_name)


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
    _check_numeric(sdfg, vsdfg, ref_args, vec_args, argnames, len_1d, kernel_name)
