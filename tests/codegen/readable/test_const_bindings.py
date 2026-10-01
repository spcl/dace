# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
A transient written once by an assignment tasklet is bound ``const`` at its write by the readable generator. The
binding is only sound where every read follows the write inside the scope that encloses it, so every other shape of
use keeps a mutable declaration.
"""
import re
from collections.abc import Callable

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg.state import LoopRegion
from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, run_isolated,
                                             use_implementation)

N = 8


def base_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_transient("x", [1], dace.float64)
    return sdfg


def bound_in_map_sdfg(name: str, wcr=None, dynamic=False, setzero=False, second_producer=False) -> dace.SDFG:
    """``x = a[i] * 2`` then ``b[i] = x + 1``, all inside one map."""
    sdfg = base_sdfg(name)
    state = sdfg.add_state()
    ra, wb = state.add_read("a"), state.add_write("b")
    entry, exit_ = state.add_map("m", {"i": f"0:{N}"})
    produce = state.add_tasklet("produce", {"v"}, {"o"}, "o = v * 2.0")
    x = state.add_access("x")
    x.setzero = setzero
    consume = state.add_tasklet("consume", {"w"}, {"o"}, "o = w + 1.0")
    state.add_memlet_path(ra, entry, produce, dst_conn="v", memlet=dace.Memlet("a[i]"))
    state.add_edge(produce, "o", x, None, dace.Memlet("x[0]", wcr=wcr, dynamic=dynamic))
    state.add_edge(x, None, consume, "w", dace.Memlet("x[0]"))
    state.add_memlet_path(consume, exit_, wb, src_conn="o", memlet=dace.Memlet("b[i]"))
    if second_producer:
        again = state.add_tasklet("again", {"v"}, {"o"}, "o = v * 3.0")
        state.add_memlet_path(ra, entry, again, dst_conn="v", memlet=dace.Memlet("a[i]"))
        state.add_edge(again, "o", x, None, dace.Memlet("x[0]"))
    return sdfg


def read_after_map_sdfg(name: str) -> dace.SDFG:
    """x is written in a map body and read after the map, outside the scope of the write."""
    sdfg = base_sdfg(name)
    state = sdfg.add_state()
    ra, wb, x = state.add_read("a"), state.add_write("b"), state.add_access("x")
    entry, exit_ = state.add_map("m", {"i": "0:1"})
    produce = state.add_tasklet("produce", {"v"}, {"o"}, "o = v * 2.0")
    state.add_memlet_path(ra, entry, produce, dst_conn="v", memlet=dace.Memlet("a[0]"))
    state.add_memlet_path(produce, exit_, x, src_conn="o", memlet=dace.Memlet("x[0]"))
    consume = state.add_tasklet("consume", {"w"}, {"o"}, "o = w + 1.0")
    state.add_edge(x, None, consume, "w", dace.Memlet("x[0]"))
    state.add_edge(consume, "o", wb, None, dace.Memlet("b[0]"))
    return sdfg


def two_state_sdfg(name: str) -> dace.SDFG:
    """x is written in one state and read in the next."""
    sdfg = base_sdfg(name)
    first = sdfg.add_state()
    produce = first.add_tasklet("produce", {"v"}, {"o"}, "o = v * 2.0")
    first.add_edge(first.add_read("a"), None, produce, "v", dace.Memlet("a[0]"))
    first.add_edge(produce, "o", first.add_access("x"), None, dace.Memlet("x[0]"))
    second = sdfg.add_state_after(first)
    consume = second.add_tasklet("consume", {"w"}, {"o"}, "o = w + 1.0")
    second.add_edge(second.add_access("x"), None, consume, "w", dace.Memlet("x[0]"))
    second.add_edge(consume, "o", second.add_write("b"), None, dace.Memlet("b[0]"))
    return sdfg


def symbolic_read_sdfg(name: str) -> dace.SDFG:
    """The value is also read by an interstate edge, which no access node shows."""
    sdfg = two_state_sdfg(name)
    sdfg.edges()[0].data.assignments["flag"] = "x[0]"
    return sdfg


def nested_sdfg_reader_sdfg(name: str) -> dace.SDFG:
    """x is read by a nested SDFG, which may take the value by non-const reference."""
    sdfg = base_sdfg(name)
    inner = dace.SDFG(f"{name}_inner")
    inner.add_array("x", [1], dace.float64)
    inner.add_array("b", [N], dace.float64)
    istate = inner.add_state()
    t = istate.add_tasklet("copy", {"w"}, {"o"}, "o = w")
    istate.add_edge(istate.add_read("x"), None, t, "w", dace.Memlet("x[0]"))
    istate.add_edge(t, "o", istate.add_write("b"), None, dace.Memlet("b[0]"))
    state = sdfg.add_state()
    produce = state.add_tasklet("produce", {"v"}, {"o"}, "o = v * 2.0")
    x = state.add_access("x")
    nested = state.add_nested_sdfg(inner, {"x"}, {"b"})
    nested.no_inline = True
    state.add_edge(state.add_read("a"), None, produce, "v", dace.Memlet("a[0]"))
    state.add_edge(produce, "o", x, None, dace.Memlet("x[0]"))
    state.add_edge(x, None, nested, "x", dace.Memlet("x[0]"))
    state.add_edge(nested, "b", state.add_write("b"), None, dace.Memlet("b[0:8]"))
    return sdfg


def nested_sdfg_writer_sdfg(name: str) -> dace.SDFG:
    """x is written by a nested SDFG instead of a tasklet."""
    sdfg = base_sdfg(name)
    inner = dace.SDFG(f"{name}_inner")
    inner.add_array("x", [1], dace.float64)
    istate = inner.add_state()
    t = istate.add_tasklet("fill", {}, {"o"}, "o = 2.0")
    istate.add_edge(t, "o", istate.add_write("x"), None, dace.Memlet("x[0]"))
    state = sdfg.add_state()
    nested = state.add_nested_sdfg(inner, {}, {"x"})
    nested.no_inline = True
    x = state.add_access("x")
    state.add_edge(nested, "x", x, None, dace.Memlet("x[0]"))
    consume = state.add_tasklet("consume", {"w"}, {"o"}, "o = w + 1.0")
    state.add_edge(x, None, consume, "w", dace.Memlet("x[0]"))
    state.add_edge(consume, "o", state.add_write("b"), None, dace.Memlet("b[0]"))
    return sdfg


def viewed_sdfg(name: str) -> dace.SDFG:
    """x is also reached through a view."""
    sdfg = bound_in_map_sdfg(name)
    sdfg.add_view("alias", [1], dace.float64)
    (state, ) = sdfg.states()
    x = next(n for n in state.data_nodes() if n.data == "x")
    view = state.add_access("alias")
    sink = state.add_tasklet("sink", {"w"}, {}, "pass")
    state.add_edge(x, None, view, "views", dace.Memlet("x[0]"))
    state.add_edge(view, None, sink, "w", dace.Memlet("alias[0]"))
    return sdfg


def storage_sdfg(name: str, lifetime=None, storage=None) -> dace.SDFG:
    sdfg = bound_in_map_sdfg(name)
    if lifetime is not None:
        sdfg.arrays["x"].lifetime = lifetime
    if storage is not None:
        sdfg.arrays["x"].storage = storage
    return sdfg


def loop_body_sdfg(name: str) -> dace.SDFG:
    """The write and the reads sit in the state of a loop body."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_scalar("x", dace.float64, transient=True)
    sdfg.add_symbol("k", dace.int32)
    loop = LoopRegion("loop", "k < 4", "k", "k = 0", "k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state("body", is_start_block=True)
    produce = state.add_tasklet("produce", {"v"}, {"o"}, "o = v * 2.0")
    x = state.add_access("x")
    consume = state.add_tasklet("consume", {"w"}, {"o"}, "o = w + 1.0")
    state.add_edge(state.add_read("a"), None, produce, "v", dace.Memlet("a[k]"))
    state.add_edge(produce, "o", x, None, dace.Memlet("x[0]"))
    state.add_edge(x, None, consume, "w", dace.Memlet("x[0]"))
    state.add_edge(consume, "o", state.add_write("b"), None, dace.Memlet("b[k]"))
    return sdfg


def code(sdfg: dace.SDFG) -> str:
    with use_implementation(EXPERIMENTAL):
        return "\n".join(obj.clean_code for obj in sdfg.generate_code() if obj.language == "cpp")


def is_bound(sdfg: dace.SDFG) -> bool:
    return re.search(r"const double x(\[1\])? ?=", code(sdfg)) is not None


BOUND = [
    pytest.param(bound_in_map_sdfg, id="written_and_read_in_one_map_body"),
    pytest.param(loop_body_sdfg, id="written_and_read_in_a_loop_body_state"),
]

UNBOUND = [
    pytest.param(lambda n: bound_in_map_sdfg(n, second_producer=True), id="written_twice"),
    pytest.param(lambda n: bound_in_map_sdfg(n, wcr="lambda p, q: p + q"), id="write_conflict_resolution"),
    pytest.param(lambda n: bound_in_map_sdfg(n, dynamic=True), id="dynamic_write"),
    pytest.param(lambda n: bound_in_map_sdfg(n, setzero=True), id="zero_filled_on_allocation"),
    pytest.param(read_after_map_sdfg, id="read_outside_the_scope_of_the_write"),
    pytest.param(two_state_sdfg, id="read_in_a_later_state"),
    pytest.param(symbolic_read_sdfg, id="read_by_an_interstate_edge"),
    pytest.param(nested_sdfg_reader_sdfg, id="read_by_a_nested_sdfg"),
    pytest.param(nested_sdfg_writer_sdfg, id="written_by_a_nested_sdfg"),
    pytest.param(viewed_sdfg, id="reached_through_a_view"),
    pytest.param(lambda n: storage_sdfg(n, lifetime=dtypes.AllocationLifetime.SDFG), id="sdfg_lifetime"),
    pytest.param(lambda n: storage_sdfg(n, storage=dtypes.StorageType.CPU_Heap), id="heap_array"),
]


@pytest.mark.parametrize("build", BOUND)
def test_a_write_once_value_read_in_its_own_scope_is_bound_const(build: Callable[[str], dace.SDFG]):
    assert is_bound(build("bound"))


@pytest.mark.parametrize("build", UNBOUND)
def test_a_value_that_may_be_seen_before_or_outside_its_write_is_not_bound_const(build: Callable[[str], dace.SDFG]):
    assert not is_bound(build("unbound"))


@pytest.mark.parametrize("build", [bound_in_map_sdfg, loop_body_sdfg, read_after_map_sdfg, two_state_sdfg])
def test_the_binding_does_not_change_the_result(build: Callable[[str], dace.SDFG]):
    a = np.random.default_rng(0).random(N)

    def run(implementation):

        def build_and_run():
            with use_implementation(implementation):
                compiled = build(f"{build.__name__}_{implementation}").compile()
            b = np.zeros(N)
            compiled(a=a.copy(), b=b)
            return {"b": b}

        return run_isolated(build_and_run)

    assert_outputs_equivalent(run(LEGACY), run(EXPERIMENTAL), "cpu", label=build.__name__)
