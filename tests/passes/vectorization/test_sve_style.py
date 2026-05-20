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
import copy

import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

N = dace.symbol("N")


@dace.program
def axpy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] + 2.0 * b[i]


@dace.program
def gather_load(src: dace.float64[N], idx: dace.int64[N], dst: dace.float64[N], scale: dace.float64):
    for i in dace.map[0:N]:
        dst[i] = src[idx[i]] * scale


@dace.program
def scatter_store(src: dace.float64[N], idx: dace.int64[N], dst: dace.float64[N], scale: dace.float64):
    for i in dace.map[0:N]:
        dst[idx[i]] = src[i] * scale


def _run_sve_vs_ref(prog, NV, num_cores, mk_inputs, output_names):
    """Build SVE-style and reference SDFGs from ``prog``, run identical
    inputs, assert bit-exact equality on every output array.
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
        assert np.allclose(ref_inputs[name], sve_inputs[name], rtol=0, atol=0), \
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
