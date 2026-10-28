# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests consume scopes lowered to maps of processing-element loops, against the dedicated consume path."""

import numpy as np
import pytest

import dace
from dace.transformation.passes.lower_consume import LowerConsumeScopes


def _run(sdfg: dace.SDFG, lower: bool, **arguments) -> None:
    sdfg.name = f"{sdfg.name}_{'lowered' if lower else 'direct'}"
    with dace.config.set_temporary("compiler", "lower_consume", value=lower):
        sdfg(**arguments)


def _fill_and_consume(
    name: str,
    num_values: int,
    num_pes: int | str,
    code: str,
    outputs: dict[str, dace.Memlet],
    containers: dict[str, tuple[list[int], dace.typeclass]],
    condition: str | None = None,
    chunksize: int = 1,
    push: bool = False,
) -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.ConsumeEntry]:
    """An SDFG that pushes ``V[0:num_values]`` into stream ``S`` and consumes it with one tasklet.

    The tasklet reads the element (or chunk) as ``s`` and writes ``outputs``; with ``push`` it may
    also push into ``S`` through connector ``sout``. ``containers`` are the arrays it writes.
    """
    sdfg = dace.SDFG(name)
    for cname, (shape, dtype) in containers.items():
        sdfg.add_array(cname, shape, dtype)
    sdfg.add_array("V", [num_values], dace.int32)
    sdfg.add_stream("S", dace.int32, transient=True)
    state = sdfg.add_state()
    stream = state.add_access("S")
    state.add_nedge(state.add_read("V"), stream, dace.Memlet(data="S", subset="0", other_subset=f"0:{num_values}"))

    entry, exit_node = state.add_consume("cons", ("p", str(num_pes)), condition, chunksize=chunksize)
    tasklet = state.add_tasklet("task", {"s"}, set(outputs) | ({"sout"} if push else set()), code)
    state.add_edge(stream, None, entry, "IN_stream", dace.Memlet.from_array("S", sdfg.arrays["S"]))
    element = dace.Memlet("S[0]") if chunksize == 1 else dace.Memlet(f"S[0:{chunksize}]")
    element.allow_oob = True
    state.add_edge(entry, "OUT_stream", tasklet, "s", element)
    for conn, memlet in outputs.items():
        exit_node.add_in_connector(f"IN_{conn}")
        exit_node.add_out_connector(f"OUT_{conn}")
        state.add_edge(tasklet, conn, exit_node, f"IN_{conn}", memlet)
        outer = dace.Memlet.from_array(memlet.data, sdfg.arrays[memlet.data])
        outer.wcr = memlet.wcr
        outer.dynamic = True
        state.add_edge(exit_node, f"OUT_{conn}", state.add_write(memlet.data), None, outer)
    if push:
        exit_node.add_in_connector("IN_S")
        exit_node.add_out_connector("OUT_S")
        state.add_edge(tasklet, "sout", exit_node, "IN_S", dace.Memlet("S[0]", dynamic=True))
        state.add_edge(exit_node, "OUT_S", state.add_write("S"), None, dace.Memlet("S[0]", dynamic=True))
    return sdfg, state, entry


def _fibonacci(n: int) -> int:
    return n if n < 2 else _fibonacci(n - 1) + _fibonacci(n - 2)


@pytest.mark.parametrize("lower", (False, True))
def test_recursive_pushes_with_several_pes(lower: bool):
    """Tasks push back into the consumed stream; any PE may pop them, and all leave at quiescence."""
    sdfg, _, _ = _fill_and_consume(
        "consume_fib",
        1,
        4,
        "if s == 1:\n    val = 1\nelif s > 1:\n    sout = s - 1\n    sout = s - 2",
        {"val": dace.Memlet("R[0]", wcr="lambda a, b: a + b", dynamic=True)},
        {"R": ([1], dace.int64)},
        push=True,
    )
    values = np.array([15], dtype=np.int32)
    result = np.zeros(1, dtype=np.int64)
    _run(sdfg, lower, V=values, R=result)
    assert result[0] == _fibonacci(15)


@pytest.mark.parametrize("lower", (False, True))
def test_body_uses_pe_index(lower: bool):
    """Each PE counts its own tasks in its own slot; the slots add up to the number of elements."""
    num_pes = 3
    sdfg, state, _ = _fill_and_consume(
        "consume_pe_index",
        100,
        num_pes,
        "cnt = 1",
        {"cnt": dace.Memlet("C[p]", wcr="lambda a, b: a + b")},
        {"C": ([num_pes], dace.int64)},
    )
    values = np.arange(100, dtype=np.int32)
    counts = np.zeros(num_pes, dtype=np.int64)
    _run(sdfg, lower, V=values, C=counts)
    assert counts.sum() == 100


@pytest.mark.parametrize("lower", (False, True))
def test_short_last_chunk(lower: bool):
    """Chunks hold up to ``chunksize`` elements; the body sees how many through ``__dace_cons_numelems``."""
    sdfg, _, _ = _fill_and_consume(
        "consume_chunks",
        10,
        1,
        "acc = 0\nfor i in range(__dace_cons_numelems):\n    acc += s[i]\ntotal = acc\nelems = __dace_cons_numelems",
        {
            "total": dace.Memlet("T[0]", wcr="lambda a, b: a + b"),
            "elems": dace.Memlet("N[0]", wcr="lambda a, b: a + b"),
        },
        {"T": ([1], dace.int64), "N": ([1], dace.int64)},
        chunksize=4,
    )
    values = np.arange(1, 11, dtype=np.int32)
    total = np.zeros(1, dtype=np.int64)
    numelems = np.zeros(1, dtype=np.int64)
    _run(sdfg, lower, V=values, T=total, N=numelems)
    assert total[0] == values.sum()
    assert numelems[0] == 10


@pytest.mark.parametrize("lower", (False, True))
def test_condition_stops_consumption(lower: bool):
    """The condition is a stop condition: one PE stops after exactly five tasks of ten queued ones."""
    sdfg, _, _ = _fill_and_consume(
        "consume_cond",
        10,
        1,
        "cnt = 1",
        {"cnt": dace.Memlet("C[0]", wcr="lambda a, b: a + b")},
        {"C": ([1], dace.int64)},
        condition="C[0] >= 5",
    )
    values = np.arange(10, dtype=np.int32)
    counts = np.zeros(1, dtype=np.int64)
    _run(sdfg, lower, V=values, C=counts)
    assert counts[0] == 5


@pytest.mark.parametrize("lower", (False, True))
def test_condition_holding_at_start(lower: bool):
    """A stop condition that holds from the start runs no task at all."""
    sdfg, _, _ = _fill_and_consume(
        "consume_cond_start",
        4,
        2,
        "cnt = 1",
        {"cnt": dace.Memlet("C[0]", wcr="lambda a, b: a + b")},
        {"C": ([1], dace.int64)},
        condition="C[0] >= 0",
    )
    values = np.arange(4, dtype=np.int32)
    counts = np.zeros(1, dtype=np.int64)
    _run(sdfg, lower, V=values, C=counts)
    assert counts[0] == 0


def test_consume_in_map():
    """Every iteration of an outer map fills and consumes a stream of its own.

    Lowered only: the dedicated consume code generator loses the enclosing map scope here.
    """
    lower = True
    sdfg = dace.SDFG("consume_in_map")
    sdfg.add_array("V", [4, 8], dace.int32)
    sdfg.add_array("R", [4], dace.int64)
    sdfg.add_stream("S", dace.int32, transient=True)
    state = sdfg.add_state()
    map_entry, map_exit = state.add_map("rows", dict(i="0:4"))
    stream = state.add_access("S")
    fill_entry, fill_exit = state.add_map("fill", dict(j="0:8"), schedule=dace.ScheduleType.Sequential)
    fill = state.add_tasklet("fill", {"inp"}, {"out"}, "out = inp")
    state.add_memlet_path(
        state.add_read("V"), map_entry, fill_entry, fill, dst_conn="inp", memlet=dace.Memlet("V[i, j]")
    )
    state.add_memlet_path(fill, fill_exit, stream, src_conn="out", memlet=dace.Memlet("S[0]"))
    entry, exit_node = state.add_consume("cons", ("p", "2"))
    tasklet = state.add_tasklet("task", {"s"}, {"r"}, "r = s")
    state.add_edge(stream, None, entry, "IN_stream", dace.Memlet("S[0]"))
    state.add_edge(entry, "OUT_stream", tasklet, "s", dace.Memlet("S[0]"))
    state.add_memlet_path(
        tasklet,
        exit_node,
        map_exit,
        state.add_write("R"),
        src_conn="r",
        memlet=dace.Memlet("R[i]", wcr="lambda a, b: a + b", dynamic=True),
    )
    values = np.arange(32, dtype=np.int32).reshape(4, 8).copy()
    result = np.zeros(4, dtype=np.int64)
    _run(sdfg, lower, V=values, R=result)
    assert np.array_equal(result, values.sum(axis=1))


def test_lowering_leaves_no_consume_scope():
    """The pass leaves maps and loops only, and the SDFG stays valid."""
    sdfg, _, _ = _fill_and_consume(
        "consume_structure",
        4,
        2,
        "if s > 1:\n    sout = s - 1\ncnt = 1",
        {"cnt": dace.Memlet("C[0]", wcr="lambda a, b: a + b")},
        {"C": ([1], dace.int64)},
        push=True,
    )
    assert LowerConsumeScopes().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    all_nodes = [n for n, _ in sdfg.all_nodes_recursive()]
    assert not any(isinstance(n, (dace.nodes.ConsumeEntry, dace.nodes.ConsumeExit)) for n in all_nodes)
    maps = [n for n in all_nodes if isinstance(n, dace.nodes.MapEntry)]
    assert len(maps) == 1 and maps[0].map.params == ["p"]
    assert any(isinstance(n, dace.sdfg.state.LoopRegion) for n in all_nodes)


if __name__ == "__main__":
    for lower in (False, True):
        test_recursive_pushes_with_several_pes(lower)
        test_body_uses_pe_index(lower)
        test_short_last_chunk(lower)
        test_condition_stops_consumption(lower)
        test_condition_holding_at_start(lower)
    test_consume_in_map()
    test_lowering_leaves_no_consume_scope()
