# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Numerical correctness of the ``gather_intrinsic`` / ``scatter_intrinsic``
``VectorizeCPU`` knobs.

Each gather/scatter kernel is vectorized with the intrinsic on (default)
and off (main-loop per-lane scalar fan) crossed with the scalar and
masked vector-remainder strategies, at a vector-width-divisible length
(64) and a non-divisible one (65, which exercises the remainder). Every
variant must match a non-transformed reference run bit-for-bit modulo
1e-10. Index arrays are the identity ``arange`` so scatters are a valid
permutation and all accesses stay in bounds.
"""
import copy

import numpy as np
import pytest

import dace
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

LEN_1D = dace.symbol("LEN_1D")


@dace.program
def s4113(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        a[ip[i]] = b[ip[i]] + c[i]


@dace.program
def s491(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
         ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        a[ip[i]] = b[i] + c[i] * d[i]


@dace.program
def s4115(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D], sum_out: dace.float64[1]):
    sum_val = 0.0
    for i in range(LEN_1D):
        sum_val = sum_val + a[i] * b[ip[i]]
    sum_out[0] = sum_val


@dace.program
def idx_table_gather(out: dace.float64[LEN_1D], table: dace.float64[LEN_1D], idx: dace.int32[LEN_1D]):
    # Non-TSVC: a plain indirect-index gather from a separate table.
    for i in range(LEN_1D):
        out[i] = table[idx[i]] * 2.0 + 1.0


_KERNELS = {
    "s4113": (s4113, ("a", "b", "c", "ip")),
    "s491": (s491, ("a", "b", "c", "d", "ip")),
    "s4115": (s4115, ("a", "b", "ip", "sum_out")),
    "idx_table_gather": (idx_table_gather, ("out", "table", "idx")),
}


def _alloc(name: str, n: int, rng: np.random.Generator):
    if name in ("ip", "idx"):
        return np.arange(n, dtype=np.int32)
    if name == "sum_out":
        return np.zeros(1, dtype=np.float64)
    return rng.random(n).astype(np.float64)


@pytest.mark.parametrize("kernel_name", list(_KERNELS))
@pytest.mark.parametrize("gather_intrinsic", [True, False])
@pytest.mark.parametrize("scatter_intrinsic", [True, False])
@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
@pytest.mark.parametrize("len_1d", [64, 65])
def test_gather_scatter_knob(kernel_name, gather_intrinsic, scatter_intrinsic, remainder_strategy, len_1d):
    prog, argnames = _KERNELS[kernel_name]
    rng = np.random.default_rng(seed=len_1d)
    ref_args = {nm: _alloc(nm, len_1d, rng) for nm in argnames}
    vec_args = {nm: arr.copy() for nm, arr in ref_args.items()}

    tag = f"{kernel_name}_g{int(gather_intrinsic)}_s{int(scatter_intrinsic)}_{remainder_strategy}_{len_1d}"
    sdfg = copy.deepcopy(prog.to_sdfg(simplify=False))
    sdfg.name = f"ref_{tag}"
    sdfg.simplify()
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"vec_{tag}"
    try:
        VectorizeCPU(vector_width=8,
                     fail_on_unvectorizable=False,
                     remainder_strategy=remainder_strategy,
                     use_fp_factor=False,
                     branch_normalization=True,
                     gather_intrinsic=gather_intrinsic,
                     scatter_intrinsic=scatter_intrinsic).apply_pass(vsdfg, {})
    except NotImplementedError as ex:
        pytest.skip(f"vectorize NotImplementedError on {tag}: {ex}")

    sdfg.compile()(**ref_args, LEN_1D=len_1d)
    vsdfg.compile()(**vec_args, LEN_1D=len_1d)

    for nm in argnames:
        diff = np.max(np.abs(ref_args[nm] - vec_args[nm]))
        assert diff < 1e-10, f"{tag}/{nm}: max abs diff = {diff}"
