# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``InterchangeLoopsIntoLaneMaps``: a level loop over block maps becomes one block map with the loop inside."""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import move_loop_into_map as mlim
from dace.transformation.passes.cpu_specialization.interchange_loops_into_lane_maps import (
    InterchangeLoopsIntoLaneMaps,
    cpu_lane_tile,
    iterator_indexes,
)

NB, NK, NL = (dace.symbol(name) for name in ("NB", "NK", "NL"))


def level_sweep() -> dace.SDFG:
    """``for k in 1..NK: t[b, :] = q[b, k - 1, :] * 2; q[b, k, :] = q[b, k, :] + t[b, :]; s[b] += q[b, k, 0]``."""
    sdfg = dace.SDFG("level_sweep")
    sdfg.add_array("q", [NB, NK, NL], dace.float64)
    sdfg.add_array("s", [NB], dace.float64)
    sdfg.add_transient("t", [NB, NL], dace.float64)
    loop = LoopRegion("levels", "k < NK", "k", "k = 1", "k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state("scale", is_start_block=True)
    first.add_mapped_tasklet(
        "scale",
        {"b": "0:NB", "l": "0:NL"},
        {"x": dace.Memlet("q[b, k - 1, l]")},
        "y = 2 * x",
        {"y": dace.Memlet("t[b, l]")},
        external_edges=True,
    )
    second = loop.add_state_after(first, "accumulate")
    t = second.add_read("t")
    q_in, q_out = second.add_read("q"), second.add_write("q")
    entry, exit_node = second.add_map("add", {"b": "0:NB", "l": "0:NL"})
    tasklet = second.add_tasklet("add", {"a", "c"}, {"o"}, "o = a + c")
    second.add_memlet_path(q_in, entry, tasklet, dst_conn="a", memlet=dace.Memlet("q[b, k, l]"))
    second.add_memlet_path(t, entry, tasklet, dst_conn="c", memlet=dace.Memlet("t[b, l]"))
    second.add_memlet_path(tasklet, exit_node, q_out, src_conn="o", memlet=dace.Memlet("q[b, k, l]"))
    third = loop.add_state_after(second, "sum")
    third.add_mapped_tasklet(
        "sum",
        {"b": "0:NB"},
        {"x": dace.Memlet("q[b, k, 0]"), "acc": dace.Memlet("s[b]")},
        "out = acc + x",
        {"out": dace.Memlet("s[b]")},
        external_edges=True,
    )
    return sdfg


def test_level_loop_moves_into_one_block_map():
    sdfg = level_sweep()
    assert InterchangeLoopsIntoLaneMaps().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    # The loop now runs inside a map over the blocks
    loop = next(
        r for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)
    )
    assert loop.sdfg is not sdfg and sdfg.start_block.entry_node(loop.sdfg.parent_nsdfg_node).map.range[0][1] == NB - 1

    rng = np.random.default_rng(0)
    q, s = rng.random((8, 6, 5)), rng.random(8)
    ref_q, ref_s = q.copy(), s.copy()
    for k in range(1, 6):
        ref_q[:, k, :] += 2 * ref_q[:, k - 1, :]
        ref_s += ref_q[:, k, 0]
    sdfg(q=q, s=s, NB=8, NK=6, NL=5)
    assert np.allclose(q, ref_q) and np.allclose(s, ref_s)


def test_a_time_step_loop_stays_outside():
    """A loop whose iterator indexes nothing re-applies the same statements to the same cells: inside the lanes it only
    grows each lane's working set (fv3_xppm), so it is left alone."""
    sdfg = dace.SDFG("time_steps")
    sdfg.add_array("q", [NB, NL], dace.float64)
    sdfg.add_array("r", [NB, NL], dace.float64)
    loop = LoopRegion("steps", "k < NK", "k", "k = 0", "k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state("twice", is_start_block=True)
    first.add_mapped_tasklet(
        "twice",
        {"b": "0:NB", "l": "0:NL"},
        {"x": dace.Memlet("q[b, l]")},
        "y = 2 * x",
        {"y": dace.Memlet("r[b, l]")},
        external_edges=True,
    )
    second = loop.add_state_after(first, "back")
    second.add_mapped_tasklet(
        "back",
        {"b": "0:NB", "l": "0:NL"},
        {"x": dace.Memlet("r[b, l]")},
        "y = x - 1",
        {"y": dace.Memlet("q[b, l]")},
        external_edges=True,
    )
    assert InterchangeLoopsIntoLaneMaps().apply_pass(sdfg, {}) is None


def test_a_loop_whose_iterator_bounds_an_inner_loop_indexes_its_cells():
    """``for k: for j in range(k): q[b, j] += 1``: ``k`` reaches the cells only through the inner bound, a
    triangular level sweep, not a time step."""
    sdfg = dace.SDFG("triangular")
    sdfg.add_array("q", [NB, NK], dace.float64)
    outer = LoopRegion("levels", "k < NK", "k", "k = 0", "k = k + 1")
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion("below", "j < k", "j", "j = 0", "j = j + 1")
    outer.add_node(inner, is_start_block=True)
    body = inner.add_state("bump", is_start_block=True)
    body.add_mapped_tasklet(
        "bump",
        {"b": "0:NB"},
        {"x": dace.Memlet("q[b, j]")},
        "y = x + 1",
        {"y": dace.Memlet("q[b, j]")},
        external_edges=True,
    )
    assert iterator_indexes(outer)


NI = dace.symbol("NI")


@dace.program
def column_steps(q: dace.float64[NI + 2, NB, NL], f: dace.float64[NI + 2, NB, NL]):
    """Time steps over stencils along ``i`` only: the ``(b, l)`` plane columns never interact."""
    for k in range(NK):
        for i, b, l in dace.map[1 : NI + 2, 0:NB, 0:NL]:
            f[i, b, l] = q[i - 1, b, l] - q[i, b, l]
        for i, b, l in dace.map[1 : NI + 1, 0:NB, 0:NL]:
            q[i, b, l] = q[i, b, l] + 0.25 * (f[i + 1, b, l] - f[i, b, l])


def test_short_columns_run_every_step_inside_a_cache_tile():
    """Six cells a column, two arrays: 128 bytes a plane point, so a tile of 256 contiguous ``l`` stays in L2 across
    the steps; the lanes are ``(b, l)`` with ``l`` stepped by the tile."""
    sdfg = column_steps.to_sdfg(simplify=True)
    sdfg.specialize({"NI": 6})
    assert cpu_lane_tile(mlim.analyze_lanes(steps_loop(sdfg), sdfg, True, True), sdfg) == 256
    assert InterchangeLoopsIntoLaneMaps().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    (lanes,) = [n for n in sdfg.start_block.nodes() if isinstance(n, dace.nodes.MapEntry)]
    assert str(lanes.map.range) == "0:NB, 0:NL:256"
    rng = np.random.default_rng(2)
    q, f = rng.random((8, 3, 300)), np.zeros((8, 3, 300))
    ref_q, ref_f = q.copy(), f.copy()
    for _ in range(4):
        ref_f[1:] = ref_q[:-1] - ref_q[1:]
        ref_q[1:-1] = ref_q[1:-1] + 0.25 * (ref_f[2:] - ref_f[1:-1])
    sdfg(q=q, f=f, NB=3, NL=300, NK=4)
    assert np.allclose(q, ref_q) and np.allclose(f, ref_f)


def test_long_columns_keep_the_time_steps_outside():
    """A column of ``NI`` cells, ``NI`` taken large: no tile fits a cache, so the steps stay outside the lanes."""
    sdfg = column_steps.to_sdfg(simplify=True)
    assert cpu_lane_tile(mlim.analyze_lanes(steps_loop(sdfg), sdfg, True, True), sdfg) is None
    assert InterchangeLoopsIntoLaneMaps().apply_pass(sdfg, {}) is None


def steps_loop(sdfg: dace.SDFG) -> LoopRegion:
    return next(b for b in sdfg.nodes() if isinstance(b, LoopRegion))


if __name__ == "__main__":
    test_level_loop_moves_into_one_block_map()
    test_a_time_step_loop_stays_outside()
    test_a_loop_whose_iterator_bounds_an_inner_loop_indexes_its_cells()
    test_short_columns_run_every_step_inside_a_cache_tile()
    test_long_columns_keep_the_time_steps_outside()
