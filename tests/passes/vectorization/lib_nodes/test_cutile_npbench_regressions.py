from dace.transformation.passes.vectorization.enums import ISA
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for the NPBench ``dace_cutile`` pipeline fixes.

Covers, per failure class:

* **cholesky2 / contour_integral** — LAPACK-only ``linalg.Cholesky`` /
  ``linalg.Solve`` library nodes now ship a ``CuPy`` expansion (a single
  Python tasklet), so :class:`CuTileSetLibraryImplementations` can lower
  them for the Python/cuTile backend, and the pass re-runs the
  ``GPU_Device -> Sequential`` demotion after expansion (expansions
  re-stamp their inner maps with the library node's ``GPU_Device``
  schedule).
* **cholesky2 (dependent range)** — a tiled map lowered as a SEQUENTIAL
  in-kernel loop (``j = i+1 : N : W``) must not index with a grid block
  id: ``cutile_tile_dim_bids`` returns ``None`` for such dims (the loop
  variable carries the base via the recovered offset) and the cuTile
  mask uses the loop variable directly.
* **correlation** — the lane-id symbol materialization emits a
  :class:`TileIota` library node instead of a raw CPP tasklet (which the
  cuTile backend rejects), and symbols referenced only in tasklet code
  become kernel parameters.
* **mlp** — ``ExpandReduceCuPy`` keeps device-resident operands on the
  device (no ``cupy.asnumpy`` round-trip into a ``GPU_Global`` output).
* **deriche** — masked tile-op memlets are marked ``allow_oob`` so the
  full-``W``-window subsets at non-divisible / single-point boundaries
  survive validation inside ``apply_gpu_transformations()``.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.tileops.nodes import TileIota
from dace.sdfg import nodes as nd
from dace.transformation.passes.vectorization.cutile_lowering import (CuTileSetLibraryImplementations,
                                                                      _demote_residual_gpu_device_maps)
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile

N = dace.symbol("N")
M = dace.symbol("M")


def _all_tasklets(sdfg: dace.SDFG):
    """Yield every Tasklet in ``sdfg``, recursively."""
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nd.Tasklet):
            yield node


def _gpu_device_map_entries(sdfg: dace.SDFG):
    """Return all GPU_Device-scheduled MapEntry nodes in ``sdfg``, recursively."""
    return [
        node for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, nd.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device
    ]


# ---------------------------------------------------------------------------
# CuPy expansions for LAPACK-only linalg nodes (cholesky2 / contour_integral)
# ---------------------------------------------------------------------------


def test_cholesky_cupy_expansion_is_python_only():
    """The linalg.Cholesky CuPy expansion must contain only Python tasklets."""
    from dace.libraries.linalg import Cholesky

    sdfg = dace.SDFG("chol_cupy_struct")
    sdfg.add_array("A", (16, 16), dace.float64)
    sdfg.add_array("B", (16, 16), dace.float64)
    state = sdfg.add_state()
    node = Cholesky("cholesky")
    node.implementation = "CuPy"
    state.add_edge(state.add_read("A"), None, node, "_a", dace.Memlet("A[0:16, 0:16]"))
    state.add_edge(node, "_b", state.add_write("B"), None, dace.Memlet("B[0:16, 0:16]"))
    sdfg.expand_library_nodes()
    tasklets = list(_all_tasklets(sdfg))
    assert tasklets, "expansion produced no tasklet"
    assert all(t.language == dtypes.Language.Python for t in tasklets)
    assert any("cupy.linalg.cholesky" in t.code.as_string for t in tasklets)


def test_solve_cupy_expansion_is_python_only():
    """The linalg.Solve CuPy expansion must contain only Python tasklets."""
    from dace.libraries.linalg import Solve

    sdfg = dace.SDFG("solve_cupy_struct")
    sdfg.add_array("A", (8, 8), dace.float64)
    sdfg.add_array("b", (8, 4), dace.float64)
    sdfg.add_array("x", (8, 4), dace.float64)
    state = sdfg.add_state()
    node = Solve("solve")
    node.implementation = "CuPy"
    state.add_edge(state.add_read("A"), None, node, "_ain", dace.Memlet("A[0:8, 0:8]"))
    state.add_edge(state.add_read("b"), None, node, "_bin", dace.Memlet("b[0:8, 0:4]"))
    state.add_edge(node, "_bout", state.add_write("x"), None, dace.Memlet("x[0:8, 0:4]"))
    sdfg.expand_library_nodes()
    tasklets = list(_all_tasklets(sdfg))
    assert tasklets, "expansion produced no tasklet"
    assert all(t.language == dtypes.Language.Python for t in tasklets)
    assert any("cupy.linalg.solve" in t.code.as_string for t in tasklets)


# ---------------------------------------------------------------------------
# Post-expansion GPU_Device demotion (cholesky2 class)
# ---------------------------------------------------------------------------


def test_demote_residual_gpu_device_maps_helper():
    """The demotion helper re-stamps GPU_Device maps to Sequential (warning
    for non-tiny volumes)."""
    sdfg = dace.SDFG("demote_helper")
    sdfg.add_array("A", (64, ), dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map("big", {"i": "0:64"}, schedule=dtypes.ScheduleType.GPU_Device)
    t = state.add_tasklet("t", {"_a"}, {"_b"}, "_b = _a")
    state.add_memlet_path(state.add_read("A"), me, t, dst_conn="_a", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(t, mx, state.add_write("A"), src_conn="_b", memlet=dace.Memlet("A[i]"))
    with pytest.warns(UserWarning, match="demoting non-tileops GPU_Device map"):
        demoted = _demote_residual_gpu_device_maps(sdfg, strict=False, pass_name="TestPass")
    assert demoted == 1
    assert not _gpu_device_map_entries(sdfg)
    assert me.map.schedule == dtypes.ScheduleType.Sequential


def test_library_expansion_reruns_demotion():
    """CuTileSetLibraryImplementations demotes GPU_Device maps introduced by
    the expansion itself (the expansion framework re-stamps a nested
    expansion's maps with the library node's GPU_Device schedule)."""
    from dace.libraries.blas.nodes.dot import Dot

    sdfg = dace.SDFG("post_expansion_demotion")
    sdfg.add_array("x", (64, ), dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("y", (64, ), dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("r", (1, ), dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Dot("dot")
    node.schedule = dtypes.ScheduleType.GPU_Device  # what apply_gpu_transformations stamps
    state.add_edge(state.add_read("x"), None, node, "_x", dace.Memlet("x[0:64]"))
    state.add_edge(state.add_read("y"), None, node, "_y", dace.Memlet("y[0:64]"))
    state.add_edge(node, "_result", state.add_write("r"), None, dace.Memlet("r[0]"))

    lowering = CuTileSetLibraryImplementations()
    # Force the map-producing 'pure' expansion (instead of the tasklet-only
    # 'CuPy' one) so a new GPU_Device map is actually introduced.
    lowering.PREFERRED_IMPLEMENTATIONS = ("pure", )
    with pytest.warns(UserWarning):
        lowering.apply_pass(sdfg, {})
    assert not _gpu_device_map_entries(sdfg), \
        "expansion-introduced GPU_Device maps must be demoted before codegen"


# ---------------------------------------------------------------------------
# Lane-id symbol materialization emits TileIota (correlation class)
# ---------------------------------------------------------------------------


def _build_lane_id_kernel(n: int) -> dace.SDFG:
    """``B[i] = A[i] + i`` — the lane-id ``ii`` must be materialized per lane."""
    sdfg = dace.SDFG("lane_id_regression")
    sdfg.add_array("A", (n, ), dace.float64)
    sdfg.add_array("B", (n, ), dace.float64)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": f"0:{n}"})
    t = state.add_tasklet("body", {"_a"}, {"_b"}, "_b = _a + ii")
    state.add_memlet_path(state.add_access("A"), me, t, dst_conn="_a", memlet=dace.Memlet("A[ii]"))
    state.add_memlet_path(t, mx, state.add_access("B"), src_conn="_b", memlet=dace.Memlet("B[ii]"))
    return sdfg


def test_lane_id_materializes_tile_iota_not_cpp_tasklet():
    """The lane-id path places a TileIota lib node (Python-compatible for
    cuTile), not a raw CPP tasklet."""
    sdfg = _build_lane_id_kernel(16)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.CUTILE,
                                         expand_tile_nodes=False)).apply_pass(sdfg, {})
    iotas = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileIota)]
    assert iotas, "lane-id symbol should be materialized via a TileIota lib node"
    cpp_tasklets = [t for t in _all_tasklets(sdfg) if t.language != dtypes.Language.Python]
    assert not cpp_tasklets, f"no raw CPP tasklets may remain: {[t.label for t in cpp_tasklets]}"


def test_lane_id_pure_path_numerics_unchanged():
    """The TileIota-based lane-id materialization matches the unvectorized
    reference on the CPU (pure) path."""
    n = 20  # non-divisible by the tile width
    rng = np.random.default_rng(0)
    a = rng.random(n)
    b_ref = np.zeros(n)
    b_vec = np.zeros(n)
    ref = _build_lane_id_kernel(n)
    ref.name = "lane_id_reg_ref"
    vec = _build_lane_id_kernel(n)
    vec.name = "lane_id_reg_vec"
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(vec, {})
    ref.compile()(A=a.copy(), B=b_ref)
    vec.compile()(A=a.copy(), B=b_vec)
    np.testing.assert_allclose(b_vec, b_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# ExpandReduceCuPy device residency (mlp class)
# ---------------------------------------------------------------------------


def test_reduce_cupy_device_resident_no_host_roundtrip():
    """With GPU_Global operands the Reduce CuPy expansion must not round-trip
    through NumPy (asnumpy into a cupy output raises 'cannot be used for
    fill' / produces host arrays)."""
    from dace.libraries.standard import Reduce

    sdfg = dace.SDFG("reduce_cupy_device")
    sdfg.add_array("A", (32, 8), dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("out", (32, ), dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    red = Reduce("reduce", wcr="lambda a, b: a + b", axes=(1, ), identity=0)
    red.implementation = "CuPy"
    state.add_edge(state.add_read("A"), None, red, "_in", dace.Memlet("A[0:32, 0:8]"))
    state.add_edge(red, "_out", state.add_write("out"), None, dace.Memlet("out[0:32]"))
    sdfg.expand_library_nodes()
    code = "\n".join(t.code.as_string for t in _all_tasklets(sdfg))
    assert "cupy.sum" in code
    assert "asnumpy" not in code, "device-resident output must stay on the device"
    assert "asarray" not in code, "device-resident input must stay on the device"


# ---------------------------------------------------------------------------
# End-to-end cuTile pipeline regressions (GPU)
# ---------------------------------------------------------------------------


@dace.program
def _chol_plus_one(A: dace.float64[N, N], B: dace.float64[N, N]):
    B[:] = np.linalg.cholesky(A) + 1.0


@pytest.mark.gpu
def test_cholesky_cupy_e2e():
    """np.linalg.cholesky + elementwise term through the full cuTile
    pipeline (cholesky2 class: CuPy lib expansion + post-expansion
    demotion)."""
    n = 100  # non-divisible by the tile width
    rng = np.random.default_rng(1)
    q = rng.random((n, n))
    a = q @ q.T + n * np.eye(n)  # symmetric positive definite
    ref = np.linalg.cholesky(a) + 1.0
    b = np.zeros((n, n))
    sdfg = _chol_plus_one.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(32, )).apply_pass(sdfg, {})
    sdfg.compile()(A=a.copy(), B=b, N=n)
    np.testing.assert_allclose(b, ref, rtol=1e-8, atol=1e-10)


@dace.program
def _solve_copy(A: dace.float64[N, N], b: dace.float64[N, M], out: dace.float64[N, M]):
    out[:] = np.linalg.solve(A, b)


@pytest.mark.gpu
def test_solve_cupy_e2e():
    """np.linalg.solve through the full cuTile pipeline (contour_integral
    class, real dtype)."""
    n, m = 40, 12
    rng = np.random.default_rng(2)
    a = rng.random((n, n)) + n * np.eye(n)
    rhs = rng.random((n, m))
    ref = np.linalg.solve(a, rhs)
    out = np.zeros((n, m))
    sdfg = _solve_copy.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(32, )).apply_pass(sdfg, {})
    sdfg.compile()(A=a.copy(), b=rhs.copy(), out=out, N=n, M=m)
    np.testing.assert_allclose(out, ref, rtol=1e-8, atol=1e-10)


@dace.program
def _triu_copy(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[i + 1:N]:
            B[i, j] = A[i, j]


@pytest.mark.gpu
def test_dependent_range_sequential_tile_loop_e2e():
    """A tiled map with a dependent range (``j = i+1 : N``) lowers to a
    sequential in-kernel loop; its loads/stores/mask must use the loop
    variable, not a grid block id (cholesky2 triu miscompile)."""
    n = 100  # non-divisible; dependent start exercises the masked tail too
    rng = np.random.default_rng(3)
    a = rng.random((n, n))
    ref = np.triu(a, k=1)
    b = np.zeros((n, n))
    sdfg = _triu_copy.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(32, )).apply_pass(sdfg, {})
    sdfg.compile()(A=a.copy(), B=b, N=n)
    np.testing.assert_allclose(b, ref, rtol=1e-12, atol=1e-12)


@dace.program
def _boundary_cols(x: dace.float64[N, M], y: dace.float64[N, M]):
    y[:, M - 1] = 0.0
    y[:, M - 2] = 3.0 * x[:, M - 1]


def test_single_point_masked_store_lowers_without_oob_error():
    """A single-point tiled dim (deriche's ``y2[:, -1] = 0``) yields a masked
    full-width tile whose memlet subset exceeds the array bound; the
    allow_oob marking must keep apply_gpu_transformations() validation
    from rejecting it."""
    sdfg = _boundary_cols.to_sdfg(simplify=False)
    # Raised InvalidSDFGEdgeError("Memlet subset out-of-bounds") before the fix.
    VectorizeCuTile(widths=(32, )).apply_pass(sdfg, {})


@pytest.mark.gpu
def test_single_point_masked_store_e2e():
    """Numerics of the deriche-class boundary-column assignment."""
    n, m = 48, 40  # m non-divisible by the tile width
    rng = np.random.default_rng(4)
    x = rng.random((n, m))
    y = rng.random((n, m))
    ref = y.copy()
    ref[:, -1] = 0.0
    ref[:, -2] = 3.0 * x[:, -1]
    sdfg = _boundary_cols.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(32, )).apply_pass(sdfg, {})
    sdfg.compile()(x=x.copy(), y=y, N=n, M=m)
    np.testing.assert_allclose(y, ref, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    test_cholesky_cupy_expansion_is_python_only()
    test_solve_cupy_expansion_is_python_only()
    test_demote_residual_gpu_device_maps_helper()
    test_library_expansion_reruns_demotion()
    test_lane_id_materializes_tile_iota_not_cpp_tasklet()
    test_lane_id_pure_path_numerics_unchanged()
    test_reduce_cupy_device_resident_no_host_roundtrip()
    test_cholesky_cupy_e2e()
    test_solve_cupy_e2e()
    test_dependent_range_sequential_tile_loop_e2e()
    test_single_point_masked_store_lowers_without_oob_error()
    test_single_point_masked_store_e2e()
