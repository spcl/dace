# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ChooseOmpSchedule`` schedules a parallel map ``guided`` exactly when its iterations do data-dependent work."""

import numpy as np
import pytest

import dace
from dace.sdfg import infer_types
from dace.transformation.passes.cpu_specialization.choose_omp_schedule import ChooseOmpSchedule
from dace.transformation.passes.parallelize_loops import ParallelizeLoops

M = dace.symbol("M")
N = dace.symbol("N")
NNZ = dace.symbol("NNZ")


@dace.program
def csr_row_sums(indptr: dace.int64[M + 1], vals: dace.float64[NNZ], out: dace.float64[M]):
    for i in range(M):
        s = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            s = s + vals[k]
        out[i] = s


@dace.program
def dense_row_sums(vals: dace.float64[M, N], out: dace.float64[M]):
    for i in range(M):
        s = 0.0
        for k in range(N):
            s = s + vals[i, k]
        out[i] = s


def scheduled(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    ParallelizeLoops().apply_pass(sdfg, {})
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    ChooseOmpSchedule().apply_pass(sdfg, {})
    return sdfg


def top_maps(sdfg: dace.SDFG) -> list[dace.nodes.MapEntry]:
    return [
        n for st in sdfg.states() for n in st.nodes() if isinstance(n, dace.nodes.MapEntry) and st.entry_node(n) is None
    ]


def test_a_map_over_csr_rows_is_scheduled_guided():
    """Each row loops over ``indptr[i]..indptr[i+1]``: uneven work per iteration."""
    sdfg = scheduled(csr_row_sums)
    (entry,) = top_maps(sdfg)
    assert entry.map.omp_schedule == dace.OMPScheduleType.Guided
    indptr = np.array([0, 3, 3, 7, 8], dtype=np.int64)
    vals = np.arange(8, dtype=np.float64)
    out = np.zeros(4)
    sdfg(indptr=indptr, vals=vals, out=out, M=4, NNZ=8)
    assert np.allclose(out, [vals[indptr[i] : indptr[i + 1]].sum() for i in range(4)])


def test_a_map_over_dense_rows_keeps_the_default_schedule():
    """Every row loops over the same ``N`` entries: static chunks are right."""
    sdfg = scheduled(dense_row_sums)
    assert all(e.map.omp_schedule == dace.OMPScheduleType.Default for e in top_maps(sdfg))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
