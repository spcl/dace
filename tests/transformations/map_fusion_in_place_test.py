# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MapFusionInPlace: two Maps over the same range fuse without privatizing the intermediate, when every container
both bodies touch is row-private (spcl/dace#2725).

Row recurrences (``p[i, j]`` from ``p[i, j - 1]``) are the case `MapFusionVertical` refuses, because the first Map
reads the intermediate it writes non-point-wise. Every case checks the values against numpy; the refusal cases check
that a pair touching another iteration's row is left alone.
"""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import MapFusionInPlace, MapFusionVertical

N = dace.symbol("N")
#: Rows and columns of the test arrays.
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
def three_column_sweeps(v: dace.float64[N, N], u: dace.float64[N, N]):
    p = np.empty((N, N), dtype=np.float64)
    q = np.empty((N, N), dtype=np.float64)
    for i in dace.map[1 : N - 1]:
        p[i, 0] = 0.0
        q[i, 0] = 1.0
        u[i, 0] = 1.0
        u[i, N - 1] = 1.0
        for j in range(1, N - 1):
            p[i, j] = -0.25 / (0.5 * p[i, j - 1] + 2.0)
    for i in dace.map[1 : N - 1]:
        for j in range(1, N - 1):
            q[i, j] = (v[i - 1, j] + v[i, j] - v[i + 1, j] - 0.5 * q[i, j - 1]) / (0.5 * p[i, j - 1] + 2.0)
    for i in dace.map[1 : N - 1]:
        for j in range(N - 2, 0, -1):
            u[i, j] = p[i, j] * u[i, j + 1] + q[i, j]


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


@dace.program
def live_output_pair(a: dace.float64[N], b: dace.float64[N], t: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        t[i] = a[i] + b[i]
    for i in dace.map[0:N]:
        c[i] = t[i] * 2.0


def rows_reference(p0: np.ndarray, q: np.ndarray) -> None:
    p = np.empty((len(p0), len(p0)))
    p[:, 0] = p0
    for j in range(1, len(p0)):
        p[:, j] = 0.5 * p[:, j - 1] + 1.0
    for j in range(1, len(p0)):
        q[:, j] = q[:, j - 1] + p[:, j - 1]


def sweeps_reference(v: np.ndarray, u: np.ndarray) -> None:
    n = v.shape[0]
    p, q = np.empty((n, n)), np.empty((n, n))
    p[1 : n - 1, 0], q[1 : n - 1, 0], u[1 : n - 1, 0], u[1 : n - 1, n - 1] = 0.0, 1.0, 1.0, 1.0
    for j in range(1, n - 1):
        p[1 : n - 1, j] = -0.25 / (0.5 * p[1 : n - 1, j - 1] + 2.0)
    for j in range(1, n - 1):
        q[1 : n - 1, j] = (v[0 : n - 2, j] + v[1 : n - 1, j] - v[2:n, j] - 0.5 * q[1 : n - 1, j - 1]) / (
            0.5 * p[1 : n - 1, j - 1] + 2.0
        )
    for j in range(n - 2, 0, -1):
        u[1 : n - 1, j] = p[1 : n - 1, j] * u[1 : n - 1, j + 1] + q[1 : n - 1, j]


def test_a_row_recurrence_and_its_reader_fuse_in_place_where_vertical_fusion_refuses():
    vertical = recurrence_then_offset_reader.to_sdfg(simplify=True)
    assert vertical.apply_transformations_repeated(MapFusionVertical) == 0
    sdfg = recurrence_then_offset_reader.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MapFusionInPlace) == 1  # the second Map's parameter is renamed
    assert top_level_maps(sdfg) == 1
    sdfg.validate()
    rng = np.random.default_rng(0)
    p0, q = rng.random(SIZE), rng.random((SIZE, SIZE))
    expected = q.copy()
    rows_reference(p0, expected)
    sdfg(p0=p0, q=q, N=SIZE)
    assert np.allclose(q, expected)


def test_three_column_sweeps_fuse_into_one_map():
    sdfg = three_column_sweeps.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MapFusionInPlace) == 2
    assert top_level_maps(sdfg) == 1
    sdfg.validate()
    rng = np.random.default_rng(1)
    v, u = rng.random((SIZE, SIZE)), rng.random((SIZE, SIZE))
    expected = u.copy()
    sweeps_reference(v, expected)
    sdfg(v=v, u=u, N=SIZE)
    assert np.allclose(u, expected)


def test_a_reader_of_another_iterations_row_is_refused():
    sdfg = reader_of_the_row_above.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MapFusionInPlace) == 0
    assert top_level_maps(sdfg) == 2


def test_a_live_intermediate_stays_written():
    sdfg = live_output_pair.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MapFusionInPlace) == 1
    assert top_level_maps(sdfg) == 1
    a, b = np.arange(8.0), np.ones(8)
    t, c = np.zeros(8), np.zeros(8)
    sdfg(a=a, b=b, t=t, c=c, N=8)
    assert np.allclose(t, a + b) and np.allclose(c, 2 * (a + b))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
