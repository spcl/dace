# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``@dace.program(facts=...)`` registers facts of the arguments' contents as data assumptions of the parsed SDFG:
per container, and as a CSR group; they serialize and ``ParallelizeLoops`` uses them without proving them."""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.symbolic import DataFact, DataFactKind, permutation
from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

M = dace.symbol("M")
N = dace.symbol("N")
NNZ = dace.symbol("NNZ")


def loop_variables(sdfg: dace.SDFG) -> set[str]:
    return {
        r.loop_variable
        for sd in sdfg.all_sdfgs_recursive()
        for r in sd.all_control_flow_regions()
        if isinstance(r, LoopRegion)
    }


@dace.program(facts={"perm": [permutation(0, M - 1), DataFact(DataFactKind.NONNEGATIVE)]})
def scatter_through_permutation(perm: dace.int64[M], val: dace.float64[M], out: dace.float64[M]):
    for r in range(M):
        out[perm[r]] = val[r] * 2.0


@dace.program(facts={"csr": ("indptr", "indices", "vals")})
def scatter_rows(
    indptr: dace.int64[M + 1], indices: dace.int64[NNZ], vals: dace.float64[NNZ], dense: dace.float64[M, N]
):
    for i in range(M):
        for k in range(indptr[i], indptr[i + 1]):
            dense[i, indices[k]] = vals[k]


@dace.program(facts={"csr": [("A_ptr", "A_idx"), ("B_ptr", "B_idx")]})
def two_pattern_matrices(
    A_ptr: dace.int64[M + 1], A_idx: dace.int64[NNZ], B_ptr: dace.int64[N + 1], B_idx: dace.int64[NNZ]
):
    A_ptr[0] = B_ptr[0]


@dace.program(facts={"missing": DataFact(DataFactKind.NONNEGATIVE)})
def names_no_argument(A: dace.float64[M]):
    A[:] = 1.0


def test_declared_facts_are_data_assumptions_that_serialize_and_map_the_scatter():
    sdfg = scatter_through_permutation.to_sdfg(simplify=True)
    assert sdfg.data_assumptions("perm") == (permutation(0, M - 1), DataFact(DataFactKind.NONNEGATIVE))
    assert dace.SDFG.from_json(sdfg.to_json()).data_assumptions("perm") == sdfg.data_assumptions("perm")
    ParallelizeLoops().apply_pass(sdfg, {})
    assert "r" not in loop_variables(sdfg)
    perm = np.random.default_rng(0).permutation(20).astype(np.int64)
    val, out = np.arange(20, dtype=np.float64), np.zeros(20)
    sdfg(perm=perm, val=val, out=out, M=20)
    ref = np.zeros(20)
    ref[perm] = val * 2.0
    assert np.array_equal(out, ref)


def test_a_csr_group_declares_nondecreasing_row_pointers_and_unique_columns_per_row():
    sdfg = scatter_rows.to_sdfg(simplify=True)
    ((ptr_fact,), (col_fact,)) = sdfg.data_assumptions("indptr"), sdfg.data_assumptions("indices")
    assert ptr_fact.kind is DataFactKind.NONDECREASING and (str(ptr_fact.lo), str(ptr_fact.hi)) == ("0", "M")
    assert col_fact == DataFact(DataFactKind.SEGMENT_INJECTIVE, segments="indptr")
    assert not sdfg.data_assumptions("vals")
    ParallelizeLoops().apply_pass(sdfg, {})
    assert not loop_variables(sdfg)
    indptr = np.array([0, 2, 2, 5], dtype=np.int64)
    indices = np.array([3, 0, 1, 4, 2], dtype=np.int64)
    vals = np.arange(5, dtype=np.float64) + 1
    dense = np.zeros((3, 5))
    sdfg(indptr=indptr, indices=indices, vals=vals, dense=dense, M=3, N=5, NNZ=5)
    ref = np.zeros((3, 5))
    for i in range(3):
        ref[i, indices[indptr[i] : indptr[i + 1]]] = vals[indptr[i] : indptr[i + 1]]
    assert np.array_equal(dense, ref)


def test_a_csr_list_declares_every_pattern_matrix():
    sdfg = two_pattern_matrices.to_sdfg(simplify=False)
    for ptr, idx, last in (("A_ptr", "A_idx", "M"), ("B_ptr", "B_idx", "N")):
        assert [(f.kind, str(f.hi)) for f in sdfg.data_assumptions(ptr)] == [(DataFactKind.NONDECREASING, last)]
        assert sdfg.data_assumptions(idx) == (DataFact(DataFactKind.SEGMENT_INJECTIVE, segments=ptr),)


def test_a_fact_of_no_argument_is_refused():
    with pytest.raises(KeyError):
        names_no_argument.to_sdfg(simplify=False)


if __name__ == "__main__":
    test_declared_facts_are_data_assumptions_that_serialize_and_map_the_scatter()
    test_a_csr_group_declares_nondecreasing_row_pointers_and_unique_columns_per_row()
    test_a_csr_list_declares_every_pattern_matrix()
    test_a_fact_of_no_argument_is_refused()
