# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``InterchangeLoopsIntoLaneMaps(target="gpu")``: a loop over kernels whose points carry nothing to each other runs
inside one kernel over every axis the kernels share."""

import numpy as np

import dace
from dace import dtypes, nodes, symbolic
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.cpu_specialization.interchange_loops_into_lane_maps import (
    InterchangeLoopsIntoLaneMaps,
)

NI, NJ, NK, T = (dace.symbol(name) for name in ("NI", "NJ", "NK", "T"))
NB, NL, NV = (dace.symbol(name) for name in ("NB", "NL", "NV"))


@dace.program
def column_sweep(q: dace.float64[NI + 2, NJ, NK], f: dace.float64[NI + 2, NJ, NK], w: dace.float64[T]):
    """A loop over stencils along ``i`` only, each over its own ``i`` range, each step with its own weight."""
    for t in range(T):
        for i, j, k in dace.map[1 : NI + 2, 0:NJ, 0:NK]:
            f[i, j, k] = q[i - 1, j, k] - q[i, j, k]
        for i, j, k in dace.map[1 : NI + 1, 0:NJ, 0:NK]:
            q[i, j, k] = q[i, j, k] + w[t] * (f[i + 1, j, k] - f[i, j, k])


def column_sweep_reference(q: np.ndarray, f: np.ndarray, w: np.ndarray) -> None:
    for weight in w:
        f[1:] = q[:-1] - q[1:]
        q[1:-1] = q[1:-1] + weight * (f[2:] - f[1:-1])


@dace.program
def row_sweep(q: dace.float64[NJ, NI + 2], f: dace.float64[NJ, NI + 2], w: dace.float64[T]):
    """The same stencil with ``i`` the contiguous axis: the threads must stay on ``i``."""
    for t in range(T):
        for j, i in dace.map[0:NJ, 1 : NI + 2]:
            f[j, i] = q[j, i - 1] - q[j, i]
        for j, i in dace.map[0:NJ, 1 : NI + 1]:
            q[j, i] = q[j, i] + w[t] * (f[j, i + 1] - f[j, i])


def top_maps(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    return [n for n in sdfg.start_block.nodes() if isinstance(n, nodes.MapEntry)] if len(sdfg.nodes()) == 1 else []


def test_weighted_steps_run_inside_one_map_over_the_plane_columns():
    sdfg = column_sweep.to_sdfg(simplify=True)
    assert InterchangeLoopsIntoLaneMaps(target="gpu").apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (lanes,) = top_maps(sdfg)
    # Both shared axes are lanes, in the maps' order; the stencil axis and the loop stay inside each lane
    assert str(lanes.map.range) == "0:NJ, 0:NK"
    loop = next(r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))
    assert loop.sdfg is not sdfg
    inner = [n for s in loop.states() for n in s.nodes() if isinstance(n, nodes.MapEntry)]
    assert sorted(str(e.map.range) for e in inner) == ["1:NI + 1, j_lane, k_lane", "1:NI + 2, j_lane, k_lane"]

    rng = np.random.default_rng(0)
    q, f = rng.random((10, 5, 6)), np.zeros((10, 5, 6))
    ref_q, ref_f = q.copy(), f.copy()
    w = np.array([0.5, 0.25, 0.125])
    column_sweep_reference(ref_q, ref_f, w)
    sdfg(q=q, f=f, w=w, NI=8, NJ=5, NK=6, T=3)
    assert np.allclose(q, ref_q) and np.allclose(f, ref_f)


def test_offloaded_sweep_is_one_kernel():
    sdfg = column_sweep.to_sdfg(simplify=True)
    InterchangeLoopsIntoLaneMaps(target="gpu").apply_pass(sdfg, {})
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    kernels = [
        n
        for n, s in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry)
        and n.map.schedule == dtypes.ScheduleType.GPU_Device
        and s.entry_node(n) is None
    ]
    assert len(kernels) == 1 and str(kernels[0].map.range) == "0:NJ, 0:NK"


def test_contiguous_stencil_axis_keeps_the_loop_outside():
    """The lanes would be ``j`` alone and ``i``, the contiguous axis, a serial walk in each thread: declined."""
    sdfg = row_sweep.to_sdfg(simplify=True)
    assert InterchangeLoopsIntoLaneMaps(target="gpu").apply_pass(sdfg, {}) is None
    assert any(isinstance(b, LoopRegion) for b in sdfg.nodes())


@dace.program
def time_steps(q: dace.float64[NI + 2, NJ, NK], f: dace.float64[NI + 2, NJ, NK]):
    """fv3_xppm's shape: a time-step loop, its iterator indexing nothing, over stencils along ``i``."""
    for t in range(T):
        for i, j, k in dace.map[1 : NI + 2, 0:NJ, 0:NK]:
            f[i, j, k] = q[i - 1, j, k] - q[i, j, k]
        for i, j, k in dace.map[1 : NI + 1, 0:NJ, 0:NK]:
            q[i, j, k] = q[i, j, k] + 0.5 * (f[i + 1, j, k] - f[i, j, k])


def test_time_steps_stay_outside_the_kernel():
    """Each thread would re-run every step over its column, its working set growing with the steps: declined."""
    sdfg = time_steps.to_sdfg(simplify=True)
    assert InterchangeLoopsIntoLaneMaps(target="gpu").apply_pass(sdfg, {}) is None
    assert any(isinstance(b, LoopRegion) for b in sdfg.nodes())


def blocked_column_physics() -> dace.SDFG:
    """CLOUDSC over blocks: a Fill and a map over ``(b, l)``, then a level loop whose maps add a species axis."""
    sdfg = dace.SDFG("blocked_column_physics")
    sdfg.add_array("q", [NB, NK, NL], dace.float64)
    sdfg.add_array("out", [NB, NL], dace.float64)
    sdfg.add_transient("acc", [NB, NV, NL], dace.float64)
    init = sdfg.add_state("init", is_start_block=True)
    fill = FillLibraryNode("zero")
    init.add_node(fill)
    init.add_edge(
        fill, FillLibraryNode.OUTPUT_CONNECTOR_NAME, init.add_write("acc"), None, dace.Memlet("acc[0:NB, 0:NV, 0:NL]")
    )
    loop = LoopRegion("levels", "k < NK", "k", "k = 0", "k = k + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    body = loop.add_state("species", is_start_block=True)
    body.add_mapped_tasklet(
        "species",
        {"b": "0:NB", "v": "0:NV", "l": "0:NL"},
        {"x": dace.Memlet("q[b, k, l]"), "a": dace.Memlet("acc[b, v, l]")},
        "y = a + (v + 1) * x",
        {"y": dace.Memlet("acc[b, v, l]")},
        external_edges=True,
    )
    final = sdfg.add_state_after(loop, "final")
    final.add_mapped_tasklet(
        "final",
        {"b": "0:NB", "l": "0:NL"},
        {"a": dace.Memlet("acc[b, 0, l]")},
        "y = a",
        {"y": dace.Memlet("out[b, l]")},
        external_edges=True,
    )
    return sdfg


def test_blocked_body_is_one_map_over_blocks_and_columns():
    sdfg = blocked_column_physics()
    assert InterchangeLoopsIntoLaneMaps(target="gpu").apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (lanes,) = top_maps(sdfg)
    assert str(lanes.map.range) == "0:NB, 0:NL"
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    kernels = [
        n
        for n, s in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry)
        and n.map.schedule == dtypes.ScheduleType.GPU_Device
        and s.entry_node(n) is None
    ]
    assert len(kernels) == 1


def test_blocked_body_numerics():
    sdfg = blocked_column_physics()
    InterchangeLoopsIntoLaneMaps(target="gpu").apply_pass(sdfg, {})
    rng = np.random.default_rng(1)
    q, out = rng.random((3, 4, 5)), np.zeros((3, 5))
    sdfg(q=q, out=out, NB=3, NK=4, NL=5, NV=2)
    assert np.allclose(out, q.sum(axis=1))


def test_small_plane_keeps_the_loop_outside():
    """Columns over two ``SMALL`` axes are a handful of threads: each launch of the plain loop spreads ``i`` across
    the device instead, so the loop stays outside."""
    sdfg = column_sweep.to_sdfg(simplify=True)
    for name in ("NJ", "NK"):
        sdfg.set_symbol_scale(name, symbolic.Scale.SMALL)
    assert InterchangeLoopsIntoLaneMaps(target="gpu").apply_pass(sdfg, {}) is None


if __name__ == "__main__":
    test_weighted_steps_run_inside_one_map_over_the_plane_columns()
    test_offloaded_sweep_is_one_kernel()
    test_contiguous_stencil_axis_keeps_the_loop_outside()
    test_time_steps_stay_outside_the_kernel()
    test_blocked_body_is_one_map_over_blocks_and_columns()
    test_blocked_body_numerics()
    test_small_plane_keeps_the_loop_outside()
