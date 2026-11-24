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


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_a_guarded_truth_store_into_one_element_is_a_logical_reduction()
    test_a_store_of_another_value_or_a_varying_element_stays_a_plain_store()
