# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A parallel CPU map that reduces over one of its axes shares out the OTHER axis (tsvc s176).

``for j: for i: a[i] += b[i + n - j - 1] * c[j]`` is one parallel map over ``(j, i)`` resolving the write
to ``a[i]`` over ``j``. Shared out by ``j``, every thread wrote every ``a[i]`` through an ``omp atomic``
(7.1 s against numba's 1.0 s); shared out by ``i``, each thread owns its elements and needs none.
"""

import numpy as np

import dace
from dace import dtypes, symbolic
from dace.sdfg import nodes
from dace.symbolic import Scale
from dace.transformation.passes.cpu_specialization.decide_cpu_parallel_axes import DecideCpuParallelAxes

N = 64


def accumulated_convolution() -> dace.SDFG:
    sdfg = dace.SDFG("cpu_reduction_axes_s176")
    for name in ("a", "b", "c"):
        sdfg.add_array(name, [2 * N], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "s176",
        {"j": f"0:{N}", "i": f"0:{N}"},
        {"bv": dace.Memlet(f"b[i + {N} - j - 1]"), "cv": dace.Memlet("c[j]")},
        "out = bv * cv",
        {"out": dace.Memlet("a[i]", wcr="lambda x, y: x + y")},
        schedule=dtypes.ScheduleType.CPU_Multicore,
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def test_the_reduced_axis_becomes_an_inner_sequential_loop_and_the_write_needs_no_atomic():
    sdfg = accumulated_convolution()
    assert DecideCpuParallelAxes().apply_pass(sdfg, {}) == 1
    maps = [
        (node.map.params, node.map.schedule)
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry)
    ]
    assert sorted(maps) == [(["i"], dtypes.ScheduleType.CPU_Multicore), (["j"], dtypes.ScheduleType.Sequential)], maps
    code = "".join(obj.clean_code for obj in sdfg.generate_code())
    assert "atomic" not in code, "the outer map writes disjoint elements, so no write needs an atomic"

    rng = np.random.default_rng(20261006)
    a, b, c = (rng.random(2 * N) for _ in range(3))
    expected = a.copy()
    for j in range(N):
        expected[:N] += b[N - j - 1 : 2 * N - j - 1] * c[j]
    sdfg(a=a, b=b, c=c)
    np.testing.assert_allclose(a, expected, rtol=1e-12)


def test_every_indexing_axis_is_shared_out_in_one_collapsed_map():
    """gem's blocks of points, reduced over atoms: ``sums[128*b + i]`` is indexed by ``b`` and ``i``. Sharing out
    ``b`` alone leaves threads idle when blocks are few; both go into one parallel map, collapsed, ``j`` inside."""
    sdfg = dace.SDFG("cpu_reduction_axes_blocks")
    sdfg.add_symbol("B", dace.int64)
    sdfg.add_array("x", ["128 * B", N], dace.float64)
    sdfg.add_array("sums", ["128 * B"], dace.float64)
    sdfg.add_state().add_mapped_tasklet(
        "block_sums",
        {"b": "0:B", "i": "0:128", "j": f"0:{N}"},
        {"v": dace.Memlet("x[128*b + i, j]")},
        "out = v",
        {"out": dace.Memlet("sums[128*b + i]", wcr="lambda x, y: x + y")},
        schedule=dtypes.ScheduleType.CPU_Multicore,
        external_edges=True,
    )
    assert DecideCpuParallelAxes().apply_pass(sdfg, {}) == 1
    maps = sorted(
        (node.map.params, node.map.schedule, node.map.collapse)
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry)
    )
    assert maps == [
        (["b", "i"], dtypes.ScheduleType.CPU_Multicore, 2),
        (["j"], dtypes.ScheduleType.Sequential, 1),
    ], maps
    code = "".join(obj.clean_code for obj in sdfg.generate_code())
    assert "collapse(2)" in code and "atomic" not in code

    x = np.random.default_rng(20261112).random((256, N))
    sums = np.ones(256)
    sdfg(x=x, sums=sums, B=2)
    np.testing.assert_allclose(sums, 1.0 + x.sum(axis=1), rtol=1e-12)


def test_a_reduction_to_one_element_is_left_to_the_openmp_reduction_clause():
    sdfg = dace.SDFG("cpu_reduction_axes_scalar")
    sdfg.add_array("a", [N, N], dace.float64)
    sdfg.add_array("s", [1], dace.float64)
    sdfg.add_state().add_mapped_tasklet(
        "total",
        {"i": f"0:{N}", "j": f"0:{N}"},
        {"v": dace.Memlet("a[i, j]")},
        "out = v",
        {"out": dace.Memlet("s[0]", wcr="lambda x, y: x + y")},
        schedule=dtypes.ScheduleType.CPU_Multicore,
        external_edges=True,
    )
    assert DecideCpuParallelAxes().apply_pass(sdfg, {}) is None


def test_a_reduction_indexed_by_a_small_axis_loops_over_it_outside_one_parallel_map():
    """ICON velocity's ``max_abs_vcfl[blk]`` over few blocks, reduced over levels and cells: sharing out the blocks
    leaves the threads idle, so the blocks become the outer loop and the reduced axes one parallel map inside, whose
    write names none of its own parameters."""
    sdfg = dace.SDFG("cpu_reduction_axes_small_index")
    sdfg.add_symbol("nblks", dace.int64, scale=Scale.SMALL)
    sdfg.add_array("a", ["nblks", N, N], dace.float64)
    sdfg.add_array("m", ["nblks"], dace.float64)
    sdfg.add_state().add_mapped_tasklet(
        "peak",
        {"b": "0:nblks", "k": f"0:{N}", "i": f"0:{N}"},
        {"v": dace.Memlet("a[b, k, i]")},
        "out = abs(v)",
        {"out": dace.Memlet("m[b]", wcr="lambda x, y: max(x, y)")},
        schedule=dtypes.ScheduleType.CPU_Multicore,
        external_edges=True,
    )
    assert DecideCpuParallelAxes().apply_pass(sdfg, {}) == 1
    maps = sorted(
        (node.map.params, node.map.schedule)
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry)
    )
    assert maps == [(["b"], dtypes.ScheduleType.Sequential), (["k", "i"], dtypes.ScheduleType.CPU_Multicore)], maps
    sdfg.openmp_array_reductions = True
    code = "".join(obj.clean_code for obj in sdfg.generate_code())
    assert "reduction(max:m[" in code and "atomic" not in code, "each block's peak is one OpenMP reduction"

    a = np.random.default_rng(7).standard_normal((3, N, N))
    m = np.full(3, -np.inf)
    sdfg(a=a, m=m, nblks=3)
    np.testing.assert_allclose(m, np.abs(a).max(axis=(1, 2)))


def per_atom_forces(name: str, read_back: bool) -> dace.SDFG:
    """``f[:] = 1`` then ``f[i, c] += x[i, k, c]`` over a map ``(i, k)`` around a map over ``c`` (examinimd)."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("x", [N, N, 3], dace.float64)
    sdfg.add_array("f", [N, 3], dace.float64)
    fill = sdfg.add_state()
    fill.add_mapped_tasklet(
        "fill", {"i": f"0:{N}", "c": "0:3"}, {}, "out = 1.0", {"out": dace.Memlet("f[i, c]")}, external_edges=True
    )
    state = sdfg.add_state_after(fill)
    pairs, pairs_exit = state.add_map("pairs", {"i": f"0:{N}", "k": f"0:{N}"})
    comps, comps_exit = state.add_map("comps", {"c": "0:3"})
    inputs = {"v": None, "g": None} if read_back else {"v": None}
    tasklet = state.add_tasklet("force", inputs, {"out": None}, "out = v + g" if read_back else "out = v")
    state.add_memlet_path(state.add_read("x"), pairs, comps, tasklet, dst_conn="v", memlet=dace.Memlet("x[i, k, c]"))
    if read_back:
        state.add_memlet_path(state.add_read("f"), pairs, comps, tasklet, dst_conn="g", memlet=dace.Memlet("f[i, c]"))
    state.add_memlet_path(
        tasklet,
        comps_exit,
        pairs_exit,
        state.add_write("f"),
        src_conn="out",
        memlet=dace.Memlet("f[i, c]", wcr="lambda a, b: a + b"),
    )
    sdfg.validate()
    return sdfg


def test_the_reduced_walk_accumulates_in_a_register_and_resolves_into_the_output_once():
    sdfg = per_atom_forces("cpu_reduction_axes_register_accumulator", read_back=False)
    assert DecideCpuParallelAxes().apply_pass(sdfg, {}) == 1
    registers = [n for n, d in sdfg.arrays.items() if d.transient and d.storage == dtypes.StorageType.Register]
    assert len(registers) == 1, sdfg.arrays
    state = list(sdfg.states())[1]
    writes = [e for e in state.edges() if e.data.data == "f" and e.data.wcr is not None]
    assert writes and all(
        isinstance(e.src, nodes.AccessNode) and e.src.data == registers[0] or isinstance(e.src, nodes.MapExit)
        for e in writes
    ), "only the accumulator resolves into f"
    sdfg.validate()

    x = np.random.default_rng(20261112).random((N, N, 3))
    f = np.zeros((N, 3))
    sdfg(x=x, f=f)
    np.testing.assert_allclose(f, 1.0 + x.sum(axis=1), rtol=1e-12)


def test_an_output_the_reduced_walk_reads_keeps_accumulating_in_memory():
    sdfg = per_atom_forces("cpu_reduction_axes_read_back", read_back=True)
    assert DecideCpuParallelAxes().apply_pass(sdfg, {}) == 1
    assert not [n for n, d in sdfg.arrays.items() if d.transient and d.storage == dtypes.StorageType.Register]


def test_a_contiguous_indexing_axis_is_shared_out_in_stripes_walked_inside_the_reduced_axis():
    """rayleigh_ritz_rotation's column argmax: ``m[j] = max_i |h[i, j]|`` shares out ``j``, the contiguous axis.
    Whole, each thread walks one column down ``i`` at stride ``N``; in stripes of ``j`` it walks ``i`` outside and
    the stripe inside, along the rows, and still owns every element it writes."""
    sdfg = dace.SDFG("cpu_reduction_axes_column_peak")
    sdfg.add_symbol(symbolic.NUM_THREADS_SYMBOL, dace.int64)
    sdfg.add_array("h", [N, 3 * N], dace.float64)
    sdfg.add_array("m", [3 * N], dace.float64)
    sdfg.add_state().add_mapped_tasklet(
        "peak",
        {"i": f"0:{N}", "j": f"0:{3 * N}"},
        {"v": dace.Memlet("h[i, j]")},
        "out = abs(v)",
        {"out": dace.Memlet("m[j]", wcr="lambda x, y: max(x, y)")},
        schedule=dtypes.ScheduleType.CPU_Multicore,
        external_edges=True,
    )
    assert DecideCpuParallelAxes().apply_pass(sdfg, {}) == 1
    entries = [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry)]
    schedules = sorted((len(e.map.params), e.map.schedule) for e in entries)
    assert schedules == [(1, dtypes.ScheduleType.CPU_Multicore), (2, dtypes.ScheduleType.Sequential)], schedules
    inner = next(e for e in entries if e.map.schedule == dtypes.ScheduleType.Sequential)
    assert inner.map.params == ["i", "j"], inner.map.params
    code = "".join(obj.clean_code for obj in sdfg.generate_code())
    assert "atomic" not in code, "each stripe is one iteration's own"

    h = np.random.default_rng(11).standard_normal((N, 3 * N))
    m = np.zeros(3 * N)
    sdfg(h=h, m=m)
    np.testing.assert_allclose(m, np.abs(h).max(axis=0))


if __name__ == "__main__":
    test_the_reduced_axis_becomes_an_inner_sequential_loop_and_the_write_needs_no_atomic()
    test_every_indexing_axis_is_shared_out_in_one_collapsed_map()
    test_a_reduction_to_one_element_is_left_to_the_openmp_reduction_clause()
    test_a_reduction_indexed_by_a_small_axis_loops_over_it_outside_one_parallel_map()
    test_the_reduced_walk_accumulates_in_a_register_and_resolves_into_the_output_once()
    test_an_output_the_reduced_walk_reads_keeps_accumulating_in_memory()
    test_a_contiguous_indexing_axis_is_shared_out_in_stripes_walked_inside_the_reduced_axis()
