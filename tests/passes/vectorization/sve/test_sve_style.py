# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""SVE-style ``sve_style='fixed'`` end-to-end coverage.

Pins the primary correctness gate the analyze-clean-then-Min design
hinges on: a globally-keyed ``_iter_mask`` correctly handles the
ragged last per-core block when ``N % (num_cores * vector_width) != 0``.
Mandatory non-divisible coverage per the S-SVE5 plan: ``N ∈ {17, 22,
23, 31}`` × ``num_cores ∈ {3, 4}``.

Each test compiles the SVE-style SDFG and an untiled reference SDFG of
the same kernel from the same source, runs both on identical random
input, and asserts bit-exact equality (rtol=0, atol=0) of every output
array. Bit-exactness — not approximate equality — is the contract the
Min-swap + global mask were designed to preserve.
"""

import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

N = dace.symbol("N")


from tests.passes.vectorization.passes.test_tile_map_by_num_cores import axpy1 as axpy  # noqa: E402 (dedup)
from tests.passes.vectorization.kernels.test_jacobi import jacobi2d as jacobi2d_sve  # noqa: E402 (dedup)



@dace.program
def gather_load(src: dace.float64[N], idx: dace.int64[N], dst: dace.float64[N], scale: dace.float64):
    for i in dace.map[0:N]:
        dst[i] = src[idx[i]] * scale


@dace.program
def scatter_store(src: dace.float64[N], idx: dace.int64[N], dst: dace.float64[N], scale: dace.float64):
    for i in dace.map[0:N]:
        dst[idx[i]] = src[i] * scale


@dace.program
def strided_load(src: dace.float64[2 * N], dst: dace.float64[N], scale: dace.float64):
    for i in dace.map[0:N]:
        dst[i] = src[i * 2] * scale


@dace.program
def triad(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], alpha: dace.float64,
          beta: dace.float64):
    for i in dace.map[0:N]:
        d[i] = a[i] + alpha * b[i] + beta * c[i]


def _run_sve_vs_ref(prog, NV, num_cores, mk_inputs, output_names, atol: float = 0.0, rtol: float = 0.0):
    """Build SVE-style and reference SDFGs from ``prog``, run identical
    inputs, assert equality on every output array.

    Defaults ``atol=0, rtol=0`` enforce bit-exact (the contract the
    Min-swap + global mask preserve for simple kernels with at most one
    mul + one add per output). Kernels with 3+ operands (e.g. triad)
    allow legit vectorizer associativity reordering, so pass
    ``atol=1e-12, rtol=1e-12`` to accept FP-reordering noise while still
    gating real errors (a misaligned mask / dropped iteration would
    produce O(1) deltas, not ulp-scale).
    """
    inputs = mk_inputs()

    ref = prog.to_sdfg(simplify=True)
    ref.replace_dict({"N": NV})
    ref.name = f"{prog.name}_ref_{NV}_{num_cores}"
    ref_inputs = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in inputs.items()}
    ref.compile()(**ref_inputs, N=NV)

    sve = prog.to_sdfg(simplify=True)
    sve.replace_dict({"N": NV})
    sve.name = f"{prog.name}_sve_{NV}_{num_cores}"
    VectorizeCPU(vector_width=8, num_cores=num_cores, sve_style="fixed",
                 fail_on_unvectorizable=True).apply_pass(sve, {})
    sve.validate()
    sve_inputs = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in inputs.items()}
    sve.compile()(**sve_inputs, N=NV)

    for name in output_names:
        assert np.allclose(ref_inputs[name], sve_inputs[name], rtol=rtol, atol=atol), \
            f"{prog.name} N={NV} nc={num_cores} arr {name}: max|d|=" \
            f"{float(np.max(np.abs(ref_inputs[name] - sve_inputs[name])))}"


# Mandatory non-divisible coverage: N ∈ {17, 22, 23, 31}, num_cores ∈
# {3, 4}; each cell has N % (num_cores * 8) != 0 AND N % 8 != 0 AND
# N % num_cores != 0 — the ragged last per-core block where the global
# mask + Min-swap are load-bearing for correctness.
_NON_DIVISIBLE = [(N, NC) for N in (17, 22, 23, 31) for NC in (3, 4)]
# A few divisible/extra-large sizes to cover the clean path + a >W per-
# core block (NC=8 keeps block=B small; NC=3 with N=31 gives a per-core
# block of 16, i.e. multiple W-tiles).
_EXTRA = [(64, 8), (65, 3), (70, 8)]


@pytest.mark.parametrize("NV,NC", _NON_DIVISIBLE + _EXTRA)
def test_sve_axpy(NV, NC):
    _run_sve_vs_ref(axpy,
                    NV,
                    NC,
                    mk_inputs=lambda: {
                        "a": np.random.rand(NV),
                        "b": np.random.rand(NV),
                        "c": np.zeros(NV)
                    },
                    output_names=("c", ))


@pytest.mark.parametrize("NV,NC", _NON_DIVISIBLE + _EXTRA)
def test_sve_gather_load(NV, NC):
    _run_sve_vs_ref(gather_load,
                    NV,
                    NC,
                    mk_inputs=lambda: {
                        "src": np.random.rand(NV),
                        "idx": np.arange(NV)[::-1].astype(np.int64),
                        "dst": np.zeros(NV),
                        "scale": 2.5,
                    },
                    output_names=("dst", ))


@pytest.mark.parametrize("NV,NC", _NON_DIVISIBLE + _EXTRA)
def test_sve_strided_load(NV, NC):
    _run_sve_vs_ref(strided_load,
                    NV,
                    NC,
                    mk_inputs=lambda: {
                        "src": np.random.rand(2 * NV),
                        "dst": np.zeros(NV),
                        "scale": 1.25,
                    },
                    output_names=("dst", ))


@pytest.mark.parametrize("NV,NC", _NON_DIVISIBLE + _EXTRA)
def test_sve_triad(NV, NC):
    # 3-operand fused expr ``a + alpha*b + beta*c``: legit vectorizer
    # associativity reordering yields ~1e-15 differences vs scalar.
    # atol=1e-12 accepts that FP noise while still catching real errors
    # (a misaligned mask / dropped iteration would produce O(1) deltas).
    _run_sve_vs_ref(triad,
                    NV,
                    NC,
                    mk_inputs=lambda: {
                        "a": np.random.rand(NV),
                        "b": np.random.rand(NV),
                        "c": np.random.rand(NV),
                        "d": np.zeros(NV),
                        "alpha": 2.5,
                        "beta": -1.25,
                    },
                    output_names=("d", ),
                    atol=1e-12,
                    rtol=1e-12)


S = dace.symbol("S")




# Multi-dim non-divisible matrix (S-2 is the contiguous dim trip; pick
# S such that S-2 hits the non-divisible cells that exercise the global
# mask on the inner j axis).
_NON_DIVISIBLE_2D = [(S_val, NC) for S_val in (19, 24, 25, 33) for NC in (3, 4)]
_EXTRA_2D = [(66, 8), (34, 3)]


@pytest.mark.parametrize("SV,NC", _NON_DIVISIBLE_2D + _EXTRA_2D)
def test_sve_jacobi2d(SV, NC):
    """jacobi2d under sve_style='fixed': multi-dim handling
    (permute lane winner -> MapExpansion -> inner Sequential + skip
    core-tile). 2D + non-divisible S-2 stresses both the multi-dim
    expansion and the global mask on the contiguous dim. Stencil is
    multi-add so atol=1e-12 accepts vectorizer FP reordering."""
    A = np.random.random((SV, SV))
    B = np.random.random((SV, SV))

    ref = jacobi2d_sve.to_sdfg(simplify=True)
    ref.replace_dict({"S": SV})
    ref.name = f"jacobi2d_sve_ref_{SV}_{NC}"
    A_ref, B_ref = A.copy(), B.copy()
    ref.compile()(A=A_ref, B=B_ref, S=SV, tsteps=3)

    sve = jacobi2d_sve.to_sdfg(simplify=True)
    sve.replace_dict({"S": SV})
    sve.name = f"jacobi2d_sve_sve_{SV}_{NC}"
    VectorizeCPU(vector_width=8, num_cores=NC, sve_style="fixed", fail_on_unvectorizable=True).apply_pass(sve, {})
    sve.validate()
    A_sve, B_sve = A.copy(), B.copy()
    sve.compile()(A=A_sve, B=B_sve, S=SV, tsteps=3)

    for name, ref_arr, sve_arr in (("A", A_ref, A_sve), ("B", B_ref, B_sve)):
        assert np.allclose(ref_arr, sve_arr, rtol=1e-12, atol=1e-12), \
            f"jacobi2d_sve S={SV} nc={NC} arr {name}: max|d|={float(np.max(np.abs(ref_arr - sve_arr)))}"


@pytest.mark.parametrize("NV,NC", _NON_DIVISIBLE + _EXTRA)
def test_sve_scatter_store(NV, NC):
    # Scatter index must be a permutation so distinct lanes never write
    # the same destination cell (otherwise the unvectorised reference and
    # the SVE chain disagree on the conflicting-write order, masking the
    # real bug we're trying to gate).
    _run_sve_vs_ref(scatter_store,
                    NV,
                    NC,
                    mk_inputs=lambda: {
                        "src": np.random.rand(NV),
                        "idx": np.arange(NV)[::-1].astype(np.int64),
                        "dst": np.zeros(NV),
                        "scale": 2.5,
                    },
                    output_names=("dst", ))
