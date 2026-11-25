# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A guarded store of one truth value into one element per row is a logical reduction (ICON's ``levmask(jb, jk) =
.TRUE.`` under the CFL clipping guard): the column loop is parallel, and the element ends ``True`` exactly when some
guard held."""

import ctypes
import os

import numpy as np

import dace
from dace.codegen.cpf import render
from dace.transformation.passes.canonicalize.constant_store_to_logical_reduction import (
    ConstantStoreToLogicalReduction,
)
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.codegen.cpf.conftest import assert_matches, build_standalone, call_standalone

NC = dace.symbol("NC")
KL = dace.symbol("KL")


@dace.program
def any_per_row(g: dace.bool_[KL, NC], flag: dace.bool_[KL], unset: dace.bool_[KL]):
    for k in range(KL):
        for c in range(NC):
            if g[k, c]:
                flag[k] = True
                unset[k] = False


def test_a_guarded_truth_store_into_one_element_is_a_logical_reduction():
    sdfg = any_per_row.to_sdfg(simplify=True)
    sdfg.name = "logical_any_per_row"
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    result = render(sdfg, language="c")
    header = [line for line in result.code.splitlines() if line.startswith("// cpf: proven")]
    assert header == ["// cpf: proven: all"], header
    assert "// unknown -- " not in result.code and "// sequential -- " not in result.code, result.code
    library = build_standalone(result.code, sdfg.name, language="c")
    rng = np.random.default_rng(4)
    kl, nc = 9, 40
    g = rng.random((kl, nc)) > 0.97
    g[3, :] = False
    for threads in (1, 3, 8):
        ctypes.CDLL("libgomp.so.1").omp_set_num_threads(threads)
        got = {"flag": np.zeros(kl, dtype=np.bool_), "unset": np.ones(kl, dtype=np.bool_)}
        call_standalone(library, result.sdfg, {**got, "g": g, "KL": kl, "NC": nc})
        assert_matches({"flag": g.any(axis=1), "unset": ~g.any(axis=1)}, got, f"{threads} threads")


def test_a_store_of_another_value_or_a_varying_element_stays_a_plain_store():
    @dace.program
    def not_logical(g: dace.bool_[NC], count: dace.int32[2], flag: dace.bool_[NC]):
        for c in range(NC):
            if g[c]:
                count[1] = 2
                flag[c] = True

    sdfg = not_logical.to_sdfg(simplify=True)
    assert ConstantStoreToLogicalReduction().apply_pass(sdfg, {}) is None
    assert all(
        edge.data.wcr is None for edge, graph in sdfg.all_edges_recursive() if isinstance(edge.data, dace.Memlet)
    )


@dace.program
def clip_and_flag(z: dace.float64[KL, NC], flag: dace.bool_[KL], peak: dace.float64[1], wlim: dace.float64):
    for k in range(KL):
        flag[k] = False
    top = 0.0
    for k in range(KL):
        for c in range(NC):
            if z[k, c] > wlim:
                flag[k] = True
                top = max(top, z[k, c])
                z[k, c] = wlim
    peak[0] = top


def test_a_logical_reduction_into_a_scalar_an_outlined_body_takes_by_pointer_compiles():
    """The guard reads ``z`` before the clip stores it, so the column loop stays one body, which the render outlines
    to a function taking the private ``flag`` copy as ``bool*``. Three faults met here: the fold assigned that pointer
    (``x = x || true``), the private copy the fold starts from was never set to ``false``, and ``top`` read ``z`` after
    the clip (scalar-to-symbol promotion of ``wlim`` dropped the edge ordering the read before the store)."""
    sdfg = clip_and_flag.to_sdfg(simplify=True)
    sdfg.name = "logical_clip_and_flag"
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    result = render(sdfg, language="c")
    library = build_standalone(result.code, sdfg.name, language="c")
    rng = np.random.default_rng(6)
    kl, nc, wlim = 7, 30, 0.9
    z = rng.random((kl, nc))
    z[2, :] = 0.5
    want = {"z": np.minimum(z, wlim), "flag": (z > wlim).any(axis=1), "peak": np.array([z[z > wlim].max()])}
    for threads in (1, 3, 8):
        ctypes.CDLL("libgomp.so.1").omp_set_num_threads(threads)
        got = {"z": z.copy(), "flag": np.zeros(kl, dtype=np.bool_), "peak": np.zeros(1)}
        call_standalone(library, result.sdfg, {**got, "wlim": wlim, "KL": kl, "NC": nc})
        assert_matches(want, got, f"{threads} threads")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_a_guarded_truth_store_into_one_element_is_a_logical_reduction()
    test_a_store_of_another_value_or_a_varying_element_stays_a_plain_store()
    test_a_logical_reduction_into_a_scalar_an_outlined_body_takes_by_pointer_compiles()
