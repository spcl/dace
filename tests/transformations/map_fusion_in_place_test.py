# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MapFusionVertical`` fuses an intermediate the first Map reads and writes other than point-wise (a row recurrence)
in place, without privatizing it, when no other iteration of the first Map writes what the second reads."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import MapFusionVertical

N = dace.symbol("N")
SIZE = 24


def top_level_maps(sdfg: dace.SDFG) -> int:
    return sum(
        1 for st in sdfg.states() for n in st.nodes() if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None
    )


@dace.program
def recurrence_then_offset_reader(p0: dace.float64[N], q: dace.float64[N, N]):
    p = np.empty((N, N), dtype=np.float64)
    for i in dace.map[0:N]:
        p[i, 0] = p0[i]
        for j in range(1, N):
            p[i, j] = 0.5 * p[i, j - 1] + 1.0
    for k in dace.map[0:N]:
        for j in range(1, N):
            q[k, j] = q[k, j - 1] + p[k, j - 1]


@dace.program
def reader_of_the_row_above(p0: dace.float64[N], q: dace.float64[N, N]):
    p = np.empty((N, N), dtype=np.float64)
    for i in dace.map[0:N]:
        p[i, 0] = p0[i]
        for j in range(1, N):
            p[i, j] = 0.5 * p[i, j - 1] + 1.0
    for i in dace.map[1:N]:
        for j in range(1, N):
            q[i, j] = q[i, j - 1] + p[i - 1, j]


def rows_reference(p0: np.ndarray, q: np.ndarray) -> None:
    p = np.empty((len(p0), len(p0)))
    p[:, 0] = p0
    for j in range(1, len(p0)):
        p[:, j] = 0.5 * p[:, j - 1] + 1.0
    for j in range(1, len(p0)):
        q[:, j] = q[:, j - 1] + p[:, j - 1]


def test_a_row_recurrence_and_its_reader_fuse_in_place():
    sdfg = recurrence_then_offset_reader.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MapFusionVertical) == 1
    assert top_level_maps(sdfg) == 1
    # Not privatized: the fused body reads and writes the original ``p``
    assert not any(name.startswith("__map_fusion_") for name in sdfg.arrays)
    sdfg.validate()
    rng = np.random.default_rng(0)
    p0, q = rng.random(SIZE), rng.random((SIZE, SIZE))
    expected = q.copy()
    rows_reference(p0, expected)
    sdfg(p0=p0, q=q, N=SIZE)
    assert np.allclose(q, expected)


def test_a_reader_of_another_iterations_row_is_refused():
    sdfg = reader_of_the_row_above.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MapFusionVertical) == 0
    assert top_level_maps(sdfg) == 2


if __name__ == "__main__":
    test_a_row_recurrence_and_its_reader_fuse_in_place()
    test_a_reader_of_another_iterations_row_is_refused()
