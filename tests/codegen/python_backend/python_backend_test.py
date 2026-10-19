# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The Python backend runs an SDFG as one Python function: the numbers match NumPy and the code is plain Python."""

import contextlib
from collections.abc import Iterator

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.codegen.exceptions import CodegenError
from dace.codegen.python.emit import generate
from dace.sdfg.state import LoopRegion

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


def one_tasklet_sdfg(code: str) -> dace.SDFG:
    """``b[0] = f(a[0])``, always under one name, so every rebuild rewrites the same module file."""
    sdfg = dace.SDFG("rebuilt_under_one_name")
    sdfg.add_array("a", [1], dace.float64)
    sdfg.add_array("b", [1], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("f", {"x"}, {"y"}, code)
    state.add_edge(state.add_read("a"), None, tasklet, "x", dace.Memlet("a[0]"))
    state.add_edge(tasklet, "y", state.add_write("b"), None, dace.Memlet("b[0]"))
    return sdfg


def test_a_rebuild_of_equal_size_runs_the_new_code():
    """Bytecode is cached by mtime and size; ``x + 1`` and ``x - 1`` rewritten in one tick have both equal."""
    a, plus, minus = np.array([5.0]), np.zeros(1), np.zeros(1)

    with python_backend():
        one_tasklet_sdfg("y = x + 1")(a=a, b=plus)
        one_tasklet_sdfg("y = x - 1")(a=a, b=minus)

    assert (plus[0], minus[0]) == (6.0, 4.0)


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


@dace.program
def scaled_by(x: dace.float64, A: dace.float64[N]):
    A[:] = A * x


def test_positional_arguments_bind_in_the_order_of_the_program_signature():
    sdfg = scaled_by.to_sdfg()
    with python_backend():
        compiled = sdfg.compile()
    a = np.ones(3)

    compiled(2.0, a, N=3)

    assert np.array_equal(a, [2.0, 2.0, 2.0])


def test_a_constant_sizes_a_transient_and_reaches_a_tasklet():
    sdfg = dace.SDFG("constant_sized")
    sdfg.add_symbol("C", dace.int64)
    sdfg.add_constant("C", np.int64(3))
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_transient("tmp", ["C"], dace.float64)
    state = sdfg.add_state()
    fill = state.add_tasklet("fill", {}, {"t": dace.float64}, "t = C * 2.0")
    tmp = state.add_access("tmp")
    state.add_edge(fill, "t", tmp, None, dace.Memlet("tmp[C - 1]"))
    copy = state.add_tasklet("copy", {"t": dace.float64}, {"o": dace.float64}, "o = t")
    state.add_edge(tmp, None, copy, "t", dace.Memlet("tmp[C - 1]"))
    state.add_edge(copy, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
    out = np.zeros(1)

    with python_backend():
        sdfg(out=out)

    assert out[0] == 6.0


def conditional_accumulate_sdfg() -> dace.SDFG:
    """``if c > 0: A[0] += 5`` through a dynamic write-conflict memlet."""
    sdfg = dace.SDFG("conditional_accumulate")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("maybe", {"c": dace.float64}, {"o": dace.float64}, "if c > 0:\n    o = 5.0")
    state.add_edge(state.add_read("C"), None, tasklet, "c", dace.Memlet("C[0]"))
    state.add_edge(
        tasklet, "o", state.add_write("A"), None, dace.Memlet("A[0]", wcr="lambda a, b: a + b", dynamic=True)
    )
    return sdfg


@pytest.mark.parametrize("condition, expected", [(-1.0, 1.0), (1.0, 6.0)])
def test_a_dynamic_accumulation_adds_only_on_the_path_that_writes(condition: float, expected: float):
    a = np.ones(1)

    with python_backend():
        conditional_accumulate_sdfg()(A=a, C=np.array([condition]))

    assert a[0] == expected


def branch_sized_sdfg() -> dace.SDFG:
    """``M = N + 1``, then ``tmp[M]`` written and read in either arm of ``if N > 5``."""
    from dace.properties import CodeBlock
    from dace.sdfg.state import ConditionalBlock, ControlFlowRegion

    sdfg = dace.SDFG("branch_sized")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("M", dace.int64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_transient("tmp", ["M"], dace.float64)
    init = sdfg.add_state("init", is_start_block=True)
    branches = ConditionalBlock("branches")
    sdfg.add_node(branches)
    sdfg.add_edge(init, branches, dace.InterstateEdge(assignments={"M": "N + 1"}))
    for condition, value in (("N > 5", 1.0), (None, 2.0)):
        arm = ControlFlowRegion(f"arm_{int(value)}", sdfg=sdfg)
        branches.add_branch(None if condition is None else CodeBlock(condition), arm)
        state = arm.add_state(f"use_{int(value)}", is_start_block=True)
        fill = state.add_tasklet("fill", {}, {"t": dace.float64}, f"t = {value}")
        tmp = state.add_access("tmp")
        state.add_edge(fill, "t", tmp, None, dace.Memlet("tmp[M - 1]"))
        copy = state.add_tasklet("copy", {"t": dace.float64}, {"o": dace.float64}, "o = t")
        state.add_edge(tmp, None, copy, "t", dace.Memlet("tmp[M - 1]"))
        state.add_edge(copy, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


@pytest.mark.parametrize("n, expected", [(9, 1.0), (2, 2.0)])
def test_a_transient_sized_by_an_assigned_symbol_exists_in_every_branch(n: int, expected: float):
    out = np.zeros(1)

    with python_backend():
        branch_sized_sdfg()(out=out, N=n)

    assert out[0] == expected


def modulo_family_sdfg() -> dace.SDFG:
    """The three modulo spellings an interstate assignment can carry, applied to a negative dividend."""
    sdfg = dace.SDFG("modulo_family")
    sdfg.add_array("out", [3], dace.int64)
    sdfg.add_symbol("n", dace.int64)
    first, second = sdfg.add_state(), sdfg.add_state()
    sdfg.add_edge(
        first,
        second,
        dace.InterstateEdge(assignments={"k": "py_mod(n - 7, 5)", "j": "c_mod(n - 7, 5)", "m": "CMod(n - 7, 5)"}),
    )
    tasklet = second.add_tasklet("t", {}, {"a": dace.int64, "b": dace.int64, "c": dace.int64}, "a = k\nb = j\nc = m")
    out = second.add_write("out")
    for index, conn in enumerate("abc"):
        second.add_edge(tasklet, conn, out, None, dace.Memlet(f"out[{index}]"))
    return sdfg


def test_python_modulo_floors_and_c_modulo_truncates_a_negative_dividend():
    out = np.zeros(3, dtype=np.int64)

    with python_backend():
        modulo_family_sdfg()(out=out, n=3)

    assert out.tolist() == [1, -4, -4]


def modulo_bounded_loops_sdfg() -> dace.SDFG:
    """Two counting loops, bounded by the floored and by C's truncating modulo of the negative ``n - 7``."""
    n = dace.symbol("n", dace.int64)
    sdfg = dace.SDFG("modulo_bounded_loops")
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("out", [2], dace.int64)
    previous = None
    for k, bound in enumerate([symbolic.PyMod(n - 7, 5), symbolic.CMod(n - 7, 5) + 5]):
        loop = LoopRegion(f"count{k}", f"i{k} < {symbolic.symstr(bound)}", f"i{k}", f"i{k} = 0", f"i{k} = i{k} + 1")
        sdfg.add_node(loop, is_start_block=previous is None)
        state = loop.add_state(f"step{k}", is_start_block=True)
        tasklet = state.add_tasklet(f"inc{k}", {"c"}, {"o"}, "o = c + 1")
        state.add_edge(state.add_read("out"), None, tasklet, "c", dace.Memlet(f"out[{k}]"))
        state.add_edge(tasklet, "o", state.add_write("out"), None, dace.Memlet(f"out[{k}]"))
        if previous is not None:
            sdfg.add_edge(previous, loop, dace.InterstateEdge())
        previous = loop
    return sdfg


def test_a_symbolic_modulo_bound_keeps_its_rounding():
    """``Mod(-4, 5)`` is 1 and C's ``-4 % 5`` is -4: one trip each, where Python's ``%`` would give six."""
    out = np.zeros(2, dtype=np.int64)

    with python_backend():
        modulo_bounded_loops_sdfg()(out=out, n=3)

    assert out.tolist() == [1, 1]


def doubler_sdfg(label: str) -> dace.SDFG:
    """``_y = 2 * _x`` over four elements, under connector names every instance shares."""
    inner = dace.SDFG(label)
    inner.add_array("_x", [4], dace.float64)
    inner.add_array("_y", [4], dace.float64)
    state = inner.add_state()
    entry, exit_node = state.add_map("double", {"k": "0:4"})
    tasklet = state.add_tasklet("twice", {"a"}, {"b"}, "b = a * 2")
    state.add_memlet_path(state.add_read("_x"), entry, tasklet, dst_conn="a", memlet=dace.Memlet("_x[k]"))
    state.add_memlet_path(tasklet, exit_node, state.add_write("_y"), src_conn="b", memlet=dace.Memlet("_y[k]"))
    return inner


def test_two_nested_sdfgs_with_one_connector_name_each_bind_their_own_arrays():
    """Emitted as they are, not inlined, as a tile kernel emits them: the second binds ``_x`` and ``_y`` anew."""
    sdfg = dace.SDFG("two_doublers")
    for name in "pqrs":
        sdfg.add_array(name, [4], dace.float64)
    state = sdfg.add_state()
    for source, target, label in (("p", "q", "first"), ("r", "s", "second")):
        nested = state.add_nested_sdfg(doubler_sdfg(label), {"_x"}, {"_y"})
        state.add_edge(state.add_read(source), None, nested, "_x", dace.Memlet(f"{source}[0:4]"))
        state.add_edge(nested, "_y", state.add_write(target), None, dace.Memlet(f"{target}[0:4]"))
    namespace: dict = {}
    exec(compile(generate(sdfg), "two_doublers", "exec"), namespace)
    p, r, q, s = np.arange(4.0), np.arange(10.0, 14.0), np.zeros(4), np.zeros(4)

    namespace["two_doublers"](p=p, q=q, r=r, s=s)

    assert q.tolist() == [0.0, 2.0, 4.0, 6.0] and s.tolist() == [20.0, 22.0, 24.0, 26.0]
