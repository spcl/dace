# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A parallel CPU map that reduces over one of its axes shares out the OTHER axis (tsvc s176).

``for j: for i: a[i] += b[i + n - j - 1] * c[j]`` is one parallel map over ``(j, i)`` resolving the write
to ``a[i]`` over ``j``. Shared out by ``j``, every thread wrote every ``a[i]`` through an ``omp atomic``
(7.1 s against numba's 1.0 s); shared out by ``i``, each thread owns its elements and needs none.
"""

import numpy as np

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.symbolic import Scale
from dace.transformation.passes.cpu_specialization.sequentialize_reduction_axes import SequentializeReductionAxes

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
    assert SequentializeReductionAxes().apply_pass(sdfg, {}) == 1
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
    assert SequentializeReductionAxes().apply_pass(sdfg, {}) is None


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
    assert SequentializeReductionAxes().apply_pass(sdfg, {}) == 1
    maps = sorted(
        (node.map.params, node.map.schedule) for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry)
    )
    assert maps == [(["b"], dtypes.ScheduleType.Sequential), (["k", "i"], dtypes.ScheduleType.CPU_Multicore)], maps
    sdfg.openmp_array_reductions = True
    code = "".join(obj.clean_code for obj in sdfg.generate_code())
    assert "reduction(max:m[" in code and "atomic" not in code, "each block's peak is one OpenMP reduction"

    a = np.random.default_rng(7).standard_normal((3, N, N))
    m = np.full(3, -np.inf)
    sdfg(a=a, m=m, nblks=3)
    np.testing.assert_allclose(m, np.abs(a).max(axis=(1, 2)))


if __name__ == "__main__":
    test_the_reduced_axis_becomes_an_inner_sequential_loop_and_the_write_needs_no_atomic()
    test_a_reduction_to_one_element_is_left_to_the_openmp_reduction_clause()
    test_a_reduction_indexed_by_a_small_axis_loops_over_it_outside_one_parallel_map()
