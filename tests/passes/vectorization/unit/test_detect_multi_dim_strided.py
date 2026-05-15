# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Detection tests for ``DetectMultiDimStridedLoad`` and ``DetectMultiDimStridedStore``.

These passes collapse the per-lane assign fan around a ``_packed`` access
node into a single ``strided_load_double`` / ``strided_store_double`` CPP
tasklet, when the per-lane multi-dim subsets linearise (via array strides)
to a fixed-stride sequence.

Each test runs the full pipeline through ``VectorizeCPU(lower_to_intrinsics=True)``,
then asserts that the emitted SDFG contains exactly one ``multi_dim_strided_{load,store}``
tasklet (i.e. the collapse fired) and that no ``assign_<i>`` Python tasklets
on the indirect array remain. Numerical equivalence is checked end-to-end
against the unvectorized reference.
"""
import copy

import dace
import numpy as np

from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

N = dace.symbol("N")


def _run_and_check(prog, arrays, params, *, has_load: bool, has_store: bool):
    """Run prog twice (scalar reference + vectorized with intrinsic lowering),
    assert the collapse fired, assert numerical equivalence."""
    ref = {k: v.copy() for k, v in arrays.items()}
    vec = {k: v.copy() for k, v in arrays.items()}

    sdfg = prog.to_sdfg(simplify=True)
    sdfg.name = f"{prog.name}_audit"
    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"{prog.name}_audit_v"

    VectorizeCPU(vector_width=8,
                 fail_on_unvectorizable=True,
                 lower_to_intrinsics=True,
                 use_fp_factor=True,
                 branch_normalization=False,
                 insert_copies=False).apply_pass(vsdfg, {})

    # Inspect emitted tasklets.
    load_count = 0
    store_count = 0
    leftover_assigns = []
    for state in vsdfg.all_states():
        for n in state.nodes():
            if isinstance(n, dace.nodes.Tasklet):
                if n.label == "multi_dim_strided_load":
                    load_count += 1
                elif n.label == "multi_dim_strided_store":
                    store_count += 1
                else:
                    # Fan-out tasklets are exactly ``assign_0`` .. ``assign_{W-1}``;
                    # the per-vec slice-store tasklet has shape ``assign_<lineno>_<W>``.
                    m = n.label.startswith("assign_") and n.label[len("assign_"):]
                    if m and m.isdigit():
                        leftover_assigns.append(n.label)
    assert load_count == (1 if has_load else 0), \
        f"expected {1 if has_load else 0} multi_dim_strided_load, got {load_count}"
    assert store_count == (1 if has_store else 0), \
        f"expected {1 if has_store else 0} multi_dim_strided_store, got {store_count}"
    assert not leftover_assigns, f"per-lane fan-out not collapsed: {leftover_assigns}"

    sdfg(**ref, **params)
    vsdfg(**vec, **params)
    for name in arrays:
        diff = np.max(np.abs(ref[name] - vec[name]))
        assert diff < 1e-12, f"{name} max abs diff = {diff}"


# --- diagonal load -----------------------------------------------------------


@dace.program
def diag_load(A: dace.float64[8 * N, 8 * N], dst: dace.float64[8 * N], scale: dace.float64):
    for i, in dace.map[0:8 * N:1]:
        dst[i] = A[i, i] * scale


def test_diagonal_load_collapses_to_strided_intrinsic():
    Nv = 64
    A = np.random.rand(Nv, Nv)
    _run_and_check(diag_load, {
        "A": A,
        "dst": np.zeros(Nv)
    }, {
        "N": Nv // 8,
        "scale": 1.5
    },
                   has_load=True,
                   has_store=False)


# --- diagonal store ----------------------------------------------------------


@dace.program
def diag_store(src: dace.float64[8 * N], A: dace.float64[8 * N, 8 * N], scale: dace.float64):
    for i, in dace.map[0:8 * N:1]:
        A[i, i] = src[i] * scale


def test_diagonal_store_collapses_to_strided_intrinsic():
    Nv = 64
    src = np.random.rand(Nv)
    _run_and_check(diag_store, {
        "src": src,
        "A": np.zeros((Nv, Nv))
    }, {
        "N": Nv // 8,
        "scale": 1.5
    },
                   has_load=False,
                   has_store=True)


# --- A[2*i, i] load ----------------------------------------------------------


@dace.program
def load_2i_i(A: dace.float64[2 * 8 * N, 8 * N], dst: dace.float64[8 * N], scale: dace.float64):
    for i, in dace.map[0:8 * N:1]:
        dst[i] = A[2 * i, i] * scale


def test_2i_i_load_collapses_to_strided_intrinsic():
    Nv = 64
    A = np.random.rand(2 * Nv, Nv)
    _run_and_check(load_2i_i, {
        "A": A,
        "dst": np.zeros(Nv)
    }, {
        "N": Nv // 8,
        "scale": 1.5
    },
                   has_load=True,
                   has_store=False)


# --- A[2*i, i] store ---------------------------------------------------------


@dace.program
def store_2i_i(src: dace.float64[8 * N], A: dace.float64[2 * 8 * N, 8 * N], scale: dace.float64):
    for i, in dace.map[0:8 * N:1]:
        A[2 * i, i] = src[i] * scale


def test_2i_i_store_collapses_to_strided_intrinsic():
    Nv = 64
    src = np.random.rand(Nv)
    _run_and_check(store_2i_i, {
        "src": src,
        "A": np.zeros((2 * Nv, Nv))
    }, {
        "N": Nv // 8,
        "scale": 1.5
    },
                   has_load=False,
                   has_store=True)


# --- A[i, 2*i] load ----------------------------------------------------------


@dace.program
def load_i_2i(A: dace.float64[8 * N, 2 * 8 * N], dst: dace.float64[8 * N], scale: dace.float64):
    for i, in dace.map[0:8 * N:1]:
        dst[i] = A[i, 2 * i] * scale


def test_i_2i_load_collapses_to_strided_intrinsic():
    Nv = 64
    A = np.random.rand(Nv, 2 * Nv)
    _run_and_check(load_i_2i, {
        "A": A,
        "dst": np.zeros(Nv)
    }, {
        "N": Nv // 8,
        "scale": 1.5
    },
                   has_load=True,
                   has_store=False)


# --- A[i, 2*i] store ---------------------------------------------------------


@dace.program
def store_i_2i(src: dace.float64[8 * N], A: dace.float64[8 * N, 2 * 8 * N], scale: dace.float64):
    for i, in dace.map[0:8 * N:1]:
        A[i, 2 * i] = src[i] * scale


def test_i_2i_store_collapses_to_strided_intrinsic():
    Nv = 64
    src = np.random.rand(Nv)
    _run_and_check(store_i_2i, {
        "src": src,
        "A": np.zeros((Nv, 2 * Nv))
    }, {
        "N": Nv // 8,
        "scale": 1.5
    },
                   has_load=False,
                   has_store=True)
