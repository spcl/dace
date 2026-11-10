# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.libraries.standard.nodes.fill import FillLibraryNode
from dace.transformation.passes.minimize_transients import MinimizeTransients


def scaled_rows_sdfg(name: str, copy_out: bool, clobber: bool = False) -> dace.SDFG:
    """``t = 2 * a`` (a map) then ``b[1, :] = t`` (copy out), or ``t = b[1, :]`` then ``a = 2 * t`` (copy in);
    ``clobber`` writes ``b[1, :]`` again where the copy-in's readers may still run."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [8], dace.float64)
    sdfg.add_array("b", [3, 8], dace.float64)
    sdfg.add_array("t", [8], dace.float64, transient=True)
    state = sdfg.add_state()
    t = state.add_access("t")
    if copy_out:
        state.add_mapped_tasklet(
            "scale",
            {"i": "0:8"},
            {"x": dace.Memlet("a[i]")},
            "y = 2 * x",
            {"y": dace.Memlet("t[i]")},
            external_edges=True,
            input_nodes={"a": state.add_read("a")},
            output_nodes={"t": t},
        )
        state.add_nedge(t, state.add_write("b"), dace.Memlet("t[0:8] -> [1, 0:8]"))
    else:
        b = state.add_read("b")
        state.add_nedge(b, t, dace.Memlet("b[1, 0:8] -> [0:8]"))
        state.add_mapped_tasklet(
            "scale",
            {"i": "0:8"},
            {"x": dace.Memlet("t[i]")},
            "y = 2 * x",
            {"y": dace.Memlet("a[i]")},
            external_edges=True,
            input_nodes={"t": t},
            output_nodes={"a": state.add_write("a")},
        )
        if clobber:
            state.add_mapped_tasklet(
                "zero", {"i": "0:8"}, {}, "y = 0", {"y": dace.Memlet("b[1, i]")}, external_edges=True
            )
    return sdfg


def run(sdfg: dace.SDFG) -> tuple[np.ndarray, np.ndarray]:
    a = np.arange(8, dtype=np.float64)
    b = np.arange(24, dtype=np.float64).reshape(3, 8)
    sdfg(a=a, b=b)
    return a, b


def test_copied_out_transient_is_its_target_region():
    sdfg = scaled_rows_sdfg("minimize_transients_copy_out", copy_out=True)
    expected = run(sdfg)
    assert MinimizeTransients().apply_pass(sdfg, {}) == 1
    assert "t" not in sdfg.arrays
    sdfg.validate()
    for got, want in zip(run(sdfg), expected):
        np.testing.assert_array_equal(got, want)


def test_copied_in_transient_reads_its_source_region():
    sdfg = scaled_rows_sdfg("minimize_transients_copy_in", copy_out=False)
    expected = run(sdfg)
    assert MinimizeTransients().apply_pass(sdfg, {}) == 1
    assert "t" not in sdfg.arrays
    sdfg.validate()
    for got, want in zip(run(sdfg), expected):
        np.testing.assert_array_equal(got, want)


def test_snapshot_of_a_region_written_concurrently_stays():
    sdfg = scaled_rows_sdfg("minimize_transients_clobber", copy_out=False, clobber=True)
    assert MinimizeTransients().apply_pass(sdfg, {}) is None
    assert "t" in sdfg.arrays


def test_transient_copied_to_two_regions_of_one_node_stays():
    sdfg = scaled_rows_sdfg("minimize_transients_two_rows", copy_out=True)
    state = sdfg.start_state
    t = next(n for n in state.data_nodes() if n.data == "t")
    b = next(n for n in state.data_nodes() if n.data == "b")
    state.add_nedge(t, b, dace.Memlet("t[0:8] -> [2, 0:8]"))
    assert MinimizeTransients().apply_pass(sdfg, {}) is None
    sdfg.validate()


def test_copy_out_next_to_another_write_of_the_target_stays():
    sdfg = scaled_rows_sdfg("minimize_transients_second_write", copy_out=True)
    state = sdfg.start_state
    t = next(n for n in state.data_nodes() if n.data == "t")
    fill = state.add_write("b")
    state.add_mapped_tasklet(
        "zero", {"i": "0:8"}, {}, "y = 0", {"y": dace.Memlet("b[0, i]")}, external_edges=True, output_nodes={"b": fill}
    )
    state.add_nedge(fill, t, dace.Memlet())
    assert MinimizeTransients().apply_pass(sdfg, {}) is None


def test_transient_computed_from_its_target_region_stays():
    """``a[1:7] = 0.5 * (a[:-2] + a[2:])``: the writes of a stencil must not overtake its reads."""
    sdfg = dace.SDFG("minimize_transients_stencil")
    sdfg.add_array("a", [8], dace.float64)
    sdfg.add_array("t", [6], dace.float64, transient=True)
    state = sdfg.add_state()
    t = state.add_access("t")
    state.add_mapped_tasklet(
        "stencil",
        {"i": "1:7"},
        {"l": dace.Memlet("a[i - 1]"), "r": dace.Memlet("a[i + 1]")},
        "y = 0.5 * (l + r)",
        {"y": dace.Memlet("t[i - 1]")},
        external_edges=True,
        output_nodes={"t": t},
    )
    state.add_nedge(t, state.add_write("a"), dace.Memlet("t[0:6] -> [1:7]"))
    assert MinimizeTransients().apply_pass(sdfg, {}) is None
    a = np.arange(8, dtype=np.float64) ** 2
    expected = a.copy()
    expected[1:7] = 0.5 * (a[:-2] + a[2:])
    sdfg(a=a)
    np.testing.assert_array_equal(a, expected)


def test_copy_in_read_after_a_second_reader_overwrites_it_stays():
    """``b[1, :]`` is overwritten by one reader of the copy while the other reader may still run."""
    sdfg = scaled_rows_sdfg("minimize_transients_reader_race", copy_out=False)
    state = sdfg.start_state
    t = next(n for n in state.data_nodes() if n.data == "t")
    fill = state.add_write("b")
    state.add_mapped_tasklet(
        "negate",
        {"i": "0:8"},
        {"x": dace.Memlet("t[i]")},
        "y = -x",
        {"y": dace.Memlet("b[1, i]")},
        external_edges=True,
        input_nodes={"t": t},
        output_nodes={"b": fill},
    )
    assert MinimizeTransients().apply_pass(sdfg, {}) is None


def test_transient_nothing_reads_leaves_no_isolated_source():
    sdfg = dace.SDFG("minimize_transients_dead_copy")
    sdfg.add_array("b", [8], dace.float64)
    sdfg.add_transient("t", [8], dace.float64)
    state = sdfg.add_state()
    state.add_nedge(state.add_read("b"), state.add_write("t"), dace.Memlet("b[0:8] -> [0:8]"))
    assert MinimizeTransients().apply_pass(sdfg, {}) == 1
    assert state.is_empty()
    sdfg.validate()


def copy_of_view_sdfg(name: str, copy_node: bool, clobber: bool = False) -> dace.SDFG:
    """``t = v.copy()`` for ``v`` the reshaping view of ``m``, by an edge or a copy node, then ``a = 2 * t``;
    ``clobber`` rewrites ``m`` where the readers of ``t`` may still run."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [8], dace.float64)
    sdfg.add_array("m", [2, 4], dace.float64)
    sdfg.add_view("v", [8], dace.float64)
    sdfg.add_transient("t", [8], dace.float64)
    state = sdfg.add_state()
    v, t = state.add_access("v"), state.add_access("t")
    state.add_edge(state.add_read("m"), None, v, "views", dace.Memlet("m[0:2, 0:4] -> [0:8]"))
    if copy_node:
        node = CopyLibraryNode("copy")
        state.add_node(node)
        state.add_edge(v, None, node, "_cpy_in", dace.Memlet("v[0:8]"))
        state.add_edge(node, "_cpy_out", t, None, dace.Memlet("t[0:8]"))
    else:
        state.add_nedge(v, t, dace.Memlet("v[0:8] -> [0:8]"))
    state.add_mapped_tasklet(
        "scale",
        {"i": "0:8"},
        {"x": dace.Memlet("t[i]")},
        "y = 2 * x",
        {"y": dace.Memlet("a[i]")},
        external_edges=True,
        input_nodes={"t": t},
        output_nodes={"a": state.add_write("a")},
    )
    if clobber:
        state.add_mapped_tasklet(
            "zero", {"i": "0:2", "j": "0:4"}, {}, "y = 0", {"y": dace.Memlet("m[i, j]")}, external_edges=True
        )
    return sdfg


@pytest.mark.parametrize("copy_node", [False, True])
def test_copy_of_a_view_reads_the_view(copy_node: bool):
    sdfg = copy_of_view_sdfg(f"minimize_transients_copy_of_view_{int(copy_node)}", copy_node)
    assert MinimizeTransients().apply_pass(sdfg, {}) == 1
    assert "t" not in sdfg.arrays
    sdfg.validate()
    a, m = np.zeros(8), np.arange(8, dtype=np.float64).reshape(2, 4)
    sdfg(a=a, m=m)
    np.testing.assert_array_equal(a, 2 * m.reshape(8))


def test_copy_of_a_view_whose_container_is_rewritten_stays():
    sdfg = copy_of_view_sdfg("minimize_transients_copy_of_clobbered_view", copy_node=True, clobber=True)
    assert MinimizeTransients().apply_pass(sdfg, {}) is None
    assert "t" in sdfg.arrays


def last_element_sdfg(name: str, fill: bool) -> dace.SDFG:
    """``t[0:8] = 5`` by a fill node or by a map, ``t[7] += a[3]``, ``out = t[7]``: only ``t[7]`` is ever read."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [8], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_transient("t", [8], dace.float64)
    init = sdfg.add_state()
    if fill:
        node = FillLibraryNode("five", value=5.0)
        init.add_node(node)
        init.add_edge(node, "_fill_out", init.add_write("t"), None, dace.Memlet("t[0:8]"))
    else:
        init.add_mapped_tasklet("five", {"i": "0:8"}, {}, "y = 5.0", {"y": dace.Memlet("t[i]")}, external_edges=True)
    update = sdfg.add_state_after(init)
    add = update.add_tasklet("add", {"x", "old"}, {"y"}, "y = old + x")
    update.add_edge(update.add_read("a"), None, add, "x", dace.Memlet("a[3]"))
    update.add_edge(update.add_read("t"), None, add, "old", dace.Memlet("t[7]"))
    update.add_edge(add, "y", update.add_write("t"), None, dace.Memlet("t[7]"))
    last = sdfg.add_state_after(update)
    last.add_nedge(last.add_read("t"), last.add_write("out"), dace.Memlet("t[7] -> [0]"))
    return sdfg


def test_array_read_at_one_element_is_that_element():
    sdfg = last_element_sdfg("minimize_transients_last_element", fill=True)
    MinimizeTransients().apply_pass(sdfg, {})
    assert tuple(sdfg.arrays["t"].shape) == (1,)
    sdfg.validate()
    a, out = np.arange(8, dtype=np.float64), np.zeros(1)
    sdfg(a=a, out=out)
    assert out[0] == 8.0


def test_array_a_map_writes_beyond_what_is_read_keeps_its_extent():
    sdfg = last_element_sdfg("minimize_transients_last_element_of_map", fill=False)
    MinimizeTransients().apply_pass(sdfg, {})
    assert tuple(sdfg.arrays["t"].shape) == (8,)


def stack_arrays_in_parallel_map_sdfg() -> dace.SDFG:
    """``out[i] = (2 + 3) * a[i, n - 1]`` per iteration of a parallel map, each term through a stack array of the
    nested SDFG; the first array is dead when the second is written."""
    body = dace.SDFG("stack_arrays_in_parallel_map_body")
    for symbol in ("i", "m", "n"):
        body.add_symbol(symbol, dace.int64)
    body.add_array("a", ["m", "n"], dace.float64)
    body.add_array("out", ["m"], dace.float64)
    stack = dace.StorageType.Register(force=True)
    body.add_array("first", ["n"], dace.float64, transient=True, storage=stack)
    body.add_array("second", ["n"], dace.float64, transient=True, storage=stack)
    body.add_scalar("term", dace.float64, transient=True)

    def scaled(name: str, factor: int, after: dace.SDFGState | None) -> dace.SDFGState:
        state = body.add_state(f"fill_{name}", is_start_block=after is None)
        if after is not None:
            body.add_edge(after, state, dace.InterstateEdge())
        state.add_mapped_tasklet(
            f"scale_{name}",
            {"j": "0:n"},
            {"x": dace.Memlet("a[i, j]")},
            f"y = {factor} * x",
            {"y": dace.Memlet(f"{name}[j]")},
            external_edges=True,
        )
        return state

    fill_first = scaled("first", 2, None)
    read_first = body.add_state_after(fill_first, "read_first")
    tasklet = read_first.add_tasklet("last", {"x"}, {"y"}, "y = x")
    read_first.add_edge(read_first.add_read("first"), None, tasklet, "x", dace.Memlet("first[n - 1]"))
    read_first.add_edge(tasklet, "y", read_first.add_write("term"), None, dace.Memlet("term[0]"))
    fill_second = scaled("second", 3, read_first)
    read_second = body.add_state_after(fill_second, "read_second")
    tasklet = read_second.add_tasklet("add", {"x", "t"}, {"y"}, "y = x + t")
    read_second.add_edge(read_second.add_read("second"), None, tasklet, "x", dace.Memlet("second[n - 1]"))
    read_second.add_edge(read_second.add_read("term"), None, tasklet, "t", dace.Memlet("term[0]"))
    read_second.add_edge(tasklet, "y", read_second.add_write("out"), None, dace.Memlet("out[i]"))

    sdfg = dace.SDFG("stack_arrays_in_parallel_map")
    sdfg.add_array("a", ["m", "n"], dace.float64)
    sdfg.add_array("out", ["m"], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("rows", {"i": "0:m"}, schedule=dace.ScheduleType.CPU_Multicore)
    mapping = {"i": "i", "m": "m", "n": "n"}
    nested = state.add_nested_sdfg(body, {"a": None}, {"out": None}, symbol_mapping=mapping)
    state.add_memlet_path(state.add_read("a"), entry, nested, dst_conn="a", memlet=dace.Memlet("a[0:m, 0:n]"))
    state.add_memlet_path(nested, exit_node, state.add_write("out"), src_conn="out", memlet=dace.Memlet("out[0:m]"))
    return sdfg


def test_stack_arrays_of_a_nest_in_a_parallel_map_share_one_buffer():
    sdfg = stack_arrays_in_parallel_map_sdfg()
    body = next(sd for sd in sdfg.all_sdfgs_recursive() if sd is not sdfg)
    assert MinimizeTransients().apply_pass(sdfg, {}) == 1
    assert sum(name in body.arrays for name in ("first", "second")) == 1
    a = np.arange(1.0, 13.0).reshape(3, 4).copy()
    out = np.zeros(3)
    sdfg(a=a, out=out, m=3, n=4)
    assert np.array_equal(out, 5 * a[:, -1])


if __name__ == "__main__":
    test_copied_out_transient_is_its_target_region()
    test_copied_in_transient_reads_its_source_region()
    test_snapshot_of_a_region_written_concurrently_stays()
    test_transient_copied_to_two_regions_of_one_node_stays()
    test_copy_out_next_to_another_write_of_the_target_stays()
    test_transient_computed_from_its_target_region_stays()
    test_copy_in_read_after_a_second_reader_overwrites_it_stays()
    test_transient_nothing_reads_leaves_no_isolated_source()
    test_copy_of_a_view_reads_the_view(False)
    test_copy_of_a_view_reads_the_view(True)
    test_copy_of_a_view_whose_container_is_rewritten_stays()
    test_array_read_at_one_element_is_that_element()
    test_array_a_map_writes_beyond_what_is_read_keeps_its_extent()
    test_stack_arrays_of_a_nest_in_a_parallel_map_share_one_buffer()
