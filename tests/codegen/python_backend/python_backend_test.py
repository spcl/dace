# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The Python backend runs an SDFG as one Python function: the numbers match NumPy and the code is plain Python."""

import contextlib
from collections.abc import Iterator

import numpy as np
import pytest

import dace
from dace.codegen.exceptions import CodegenError

N = dace.symbol("N")


@contextlib.contextmanager
def python_backend() -> Iterator[None]:
    with dace.config.set_temporary("compiler", "backend", value="python"):
        yield


def function_source(sdfg: dace.SDFG) -> str:
    """The generated function, without the module prelude."""
    with python_backend():
        compiled = sdfg.compile()
    source = compiled.path.read_text()
    return source[source.index(f"def {sdfg.name}(") :]


@dace.program
def scaled_accumulate(A: dace.float64[N], B: dace.float64[N], s: dace.float64):
    for t in range(3):
        for i in dace.map[0:N]:
            B[i] = B[i] + s * A[i]
        if t == 1:
            B[0] = B[0] + 100.0


def test_loops_maps_and_branches_compute_what_numpy_does():
    a, b = np.arange(5.0), np.ones(5)

    with python_backend():
        scaled_accumulate(a, b, 2.0)

    assert np.array_equal(b, [101.0, 7.0, 13.0, 19.0, 25.0])


def test_a_counted_loop_is_a_python_for_over_a_range():
    source = function_source(scaled_accumulate.to_sdfg())

    assert "while" not in source
    assert "in range(0, 3):" in source
    assert "for i in range(0, N):" in source


@dace.program
def row_sums(A: dace.float64[N, 4], S: dace.float64[N]):
    for i, j in dace.map[0:N, 0:4]:
        S[i] += A[i, j]


def test_a_write_conflict_accumulates_every_iteration():
    a = np.arange(12.0).reshape(3, 4)
    s = np.zeros(3)

    with python_backend():
        row_sums(a, s)

    assert np.array_equal(s, [6.0, 22.0, 38.0])


@dace.program
def first_negative(A: dace.float64[N], out: dace.int64[1]):
    out[0] = -1
    i = 0
    while i < N:
        if A[i] < 0:
            out[0] = i
            break
        i += 1


def test_a_while_loop_with_a_break_stops_at_the_first_hit():
    out = np.zeros(1, dtype=np.int64)

    with python_backend():
        first_negative(np.array([3.0, 1.0, -2.0, -5.0]), out)

    assert out[0] == 2


@dace.program
def skip_odd(A: dace.float64[N]):
    for i in range(N):
        if i % 2 == 1:
            continue
        A[i] = 0.0


def test_a_continue_skips_the_rest_of_the_iteration():
    a = np.ones(6)

    with python_backend():
        skip_odd(a)

    assert np.array_equal(a, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


@dace.program
def two_results(A: dace.float64[N]):
    return A + 1.0, A * 2.0


def test_several_return_values_come_back_as_a_tuple():
    a = np.arange(4.0)

    with python_backend():
        plus, times = two_results(a)

    assert np.array_equal(plus, a + 1.0) and np.array_equal(times, a * 2.0)


@dace.program
def column_view(A: dace.float64[4, 4]):
    col = A[:, 1]
    col[:] = 7.0


def test_a_view_writes_through_to_the_viewed_array():
    sdfg = column_view.to_sdfg(simplify=False)
    a = np.zeros((4, 4))

    with python_backend():
        sdfg(A=a)

    expected = np.zeros((4, 4))
    expected[:, 1] = 7.0
    assert np.array_equal(a, expected)
    assert "as_strided(A[" in function_source(sdfg)


def nested_sdfg() -> dace.SDFG:
    """An outer map over rows whose body is a nested SDFG kept apart (``no_inline``) that doubles row ``i``."""
    inner = dace.SDFG("double_row")
    inner.add_symbol("i", dace.int64)
    inner.add_array("row", [3, 4], dace.float64)
    state = inner.add_state()
    entry, exit_node = state.add_map("cols", {"j": "0:4"})
    tasklet = state.add_tasklet("double", {"x": dace.float64}, {"y": dace.float64}, "y = 2 * x")
    state.add_memlet_path(state.add_read("row"), entry, tasklet, dst_conn="x", memlet=dace.Memlet("row[i, j]"))
    state.add_memlet_path(tasklet, exit_node, state.add_write("row"), src_conn="y", memlet=dace.Memlet("row[i, j]"))

    outer = dace.SDFG("rows_of_double")
    outer.add_array("A", [3, 4], dace.float64)
    state = outer.add_state()
    entry, exit_node = state.add_map("rows", {"i": "0:3"})
    node = state.add_nested_sdfg(inner, {"row"}, {"row"}, symbol_mapping={"i": "i"})
    node.no_inline = True
    state.add_memlet_path(state.add_read("A"), entry, node, dst_conn="row", memlet=dace.Memlet("A[i, 0:4]"))
    state.add_memlet_path(node, exit_node, state.add_write("A"), src_conn="row", memlet=dace.Memlet("A[i, 0:4]"))
    return outer


def test_a_nested_sdfg_binds_its_row_in_place():
    sdfg = nested_sdfg()
    a = np.arange(12.0).reshape(3, 4)

    with python_backend():
        sdfg(A=a)

    assert np.array_equal(a, 2 * np.arange(12.0).reshape(3, 4))
    assert "row = A" in function_source(nested_sdfg())


def test_a_cpp_tasklet_is_refused():
    sdfg = dace.SDFG("cpp_tasklet")
    sdfg.add_array("A", [1], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("t", {}, {"o": dace.float64}, "o = 1.0;", language=dace.Language.CPP)
    state.add_edge(tasklet, "o", state.add_write("A"), None, dace.Memlet("A[0]"))

    with python_backend(), pytest.raises(CodegenError, match="tasklet 't' is CPP"):
        sdfg.compile()


def test_a_gpu_map_without_tile_ops_is_refused():
    sdfg = dace.SDFG("gpu_map")
    sdfg.add_array("A", [8], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "fill",
        {"i": "0:8"},
        {},
        "o = 1.0",
        {"o": dace.Memlet("A[i]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )

    with python_backend(), pytest.raises(CodegenError, match="holds no tile ops"):
        sdfg.compile()
