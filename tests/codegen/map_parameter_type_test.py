# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that CPU map loops declare their parameters with the inferred iteration symbol type."""

import re

import numpy as np

import dace
from dace import subsets, symbolic
from dace.transformation.dataflow import MapTiling


def _map_sdfg(name: str, bound_type: dace.typeclass, begin, end, step=1, size=None) -> dace.SDFG:
    """A single map over ``[begin, end]`` with ``step`` that writes each ``i`` to ``A[(i - begin) // step]``."""
    sdfg = dace.SDFG(name)
    sdfg.add_symbol("N", bound_type)
    sdfg.add_symbol("S", bound_type)
    sdfg.add_array("A", [size if size is not None else "N"], dace.int64)
    state = sdfg.add_state()
    me, mx = state.add_map("m", {"i": subsets.Range([(begin, end, step)])})
    tasklet = state.add_tasklet("t", {}, {"o"}, "o = i")
    state.add_edge(me, None, tasklet, None, dace.Memlet())
    state.add_edge(tasklet, "o", mx, None, dace.Memlet(f"A[(i - ({begin})) // ({step})]"))
    state.add_edge(mx, None, state.add_write("A"), None, dace.Memlet.from_array("A", sdfg.arrays["A"]))
    return sdfg


def _loop_headers(sdfg: dace.SDFG, param: str = "i") -> list[str]:
    code = sdfg.generate_code()[0].clean_code
    return [line.strip() for line in code.splitlines() if re.search(rf"for \([^;]*\b{param} = ", line)]


def test_default_symbol_bound_declares_int():
    sdfg = _map_sdfg("map_param_type_int32", dace.int32, 0, "N - 1")
    headers = _loop_headers(sdfg)
    assert headers and all(h.startswith("for (int i = 0;") for h in headers), headers


def test_64bit_symbol_bound_declares_int64():
    """A literal ``0`` start must not narrow the parameter to ``int`` when the end is 64-bit."""
    sdfg = _map_sdfg("map_param_type_int64", dace.int64, 0, "N - 1")
    headers = _loop_headers(sdfg)
    assert headers and all(h.startswith("for (int64_t i = 0;") for h in headers), headers


def test_out_of_range_literal_bound_declares_int64():
    sdfg = _map_sdfg("map_param_type_literal", dace.int32, 0, 2**32)
    headers = _loop_headers(sdfg)
    assert headers and all(h.startswith("for (int64_t i = 0;") for h in headers), headers


def test_negative_start_with_out_of_range_literal_stays_signed():
    """A literal beyond 32 bits is signed 64-bit, as in C; an unsigned iterate would wrap its start of -1."""
    sdfg = _map_sdfg("map_param_type_negative", dace.int32, -1, 2**32)
    headers = _loop_headers(sdfg)
    assert headers and all(h.startswith("for (int64_t i = -1;") for h in headers), headers


def test_integer_function_bound_keeps_argument_type():
    sdfg = _map_sdfg("map_param_type_int_floor", dace.uint64, 0, "int_floor(N, 2) - 1")
    headers = _loop_headers(sdfg)
    assert headers and all(h.startswith("for (uint64_t i = 0;") for h in headers), headers


def test_tiled_map_keeps_default_type():
    """A tiled map's inner end is a SymExpr (``Min(N - 1, tile_i + 31)``); its literals must not widen the iterate."""
    sdfg = _map_sdfg("map_param_type_tiled", dace.int32, 0, "N - 1")
    map_entry = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))
    MapTiling.apply_to(sdfg, options={"tile_sizes": (32,)}, map_entry=map_entry)
    assert any(
        isinstance(rng[1], symbolic.SymExpr)
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry)
        for rng in n.map.range
    )

    headers = _loop_headers(sdfg) + _loop_headers(sdfg, "tile_i")
    assert len(headers) == 2 and all(h.startswith("for (int ") for h in headers), headers


def test_specialized_symbol_bound_keeps_its_type():
    """Specializing a symbol makes it a constant, which a range bound then names (auto-optimization also drops the
    symbol itself). After a serialization round trip, the symbol instance in the bound carries the default type, so
    the constant's declared type has to decide."""
    sdfg = _map_sdfg("map_param_type_specialized", dace.int64, 0, "N - 1")
    sdfg.specialize({"N": 2**33})
    sdfg.remove_symbol("N")
    sdfg = dace.SDFG.from_json(sdfg.to_json())
    headers = _loop_headers(sdfg)
    assert headers and all(h.startswith("for (int64_t i = 0;") for h in headers), headers


def test_dynamic_map_range_declares_connector_type():
    sdfg = dace.SDFG("map_param_type_dynamic_range")
    sdfg.add_array("A", [10], dace.int64)
    sdfg.add_scalar("n", dace.int64)
    state = sdfg.add_state()
    me, mx = state.add_map("m", {"i": subsets.Range([(0, "n - 1", 1)])})
    me.add_in_connector("n", dace.int64)
    state.add_edge(state.add_read("n"), None, me, "n", dace.Memlet("n[0]"))
    tasklet = state.add_tasklet("t", {}, {"o"}, "o = i")
    state.add_edge(me, None, tasklet, None, dace.Memlet())
    state.add_edge(tasklet, "o", mx, None, dace.Memlet("A[i]"))
    state.add_edge(mx, None, state.add_write("A"), None, dace.Memlet("A[0:10]"))
    headers = _loop_headers(sdfg)
    assert headers and all(h.startswith("for (int64_t i = 0;") for h in headers), headers


def test_64bit_iteration_does_not_overflow():
    """Stepping a parameter past 2**31 requires the 64-bit declaration: an ``int`` would overflow."""
    step = 2**30
    count = 5
    sdfg = _map_sdfg("map_param_type_overflow", dace.int64, 0, "N - 1", "S", size=count)
    A = np.full([count], -1, dtype=np.int64)
    sdfg(A=A, N=count * step, S=step)
    assert np.array_equal(A, np.arange(count, dtype=np.int64) * step), A


N64 = dace.symbol("N64", dace.int64)
S64 = dace.symbol("S64", dace.int64)
M32 = dace.symbol("M32", dace.int32)


@dace.program
def strided_conditional(out: dace.int64[8]):
    for i in dace.map[0:N64:S64]:
        if i >= 0:
            out[i // S64] = i


@dace.program
def default_conditional(out: dace.int64[M32]):
    for i in dace.map[0:M32]:
        if i >= 0:
            out[i] = i


def _nested_symbol_types(sdfg: dace.SDFG, name: str) -> list[dace.typeclass]:
    return [n.sdfg.symbols[name] for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]


def test_frontend_declares_map_parameter_with_its_range_type():
    """The body of a map is a nested SDFG, which receives the parameter as a symbol of the type the frontend infers
    from the range (``start:stop:step``); it must match the map's, or the call narrows the parameter."""
    assert _nested_symbol_types(strided_conditional.to_sdfg(simplify=False), "i") == [dace.int64]
    assert _nested_symbol_types(default_conditional.to_sdfg(simplify=False), "i") == [dace.int32]


def test_frontend_map_parameter_beyond_32_bits():
    step = 2**30
    out = np.zeros(8, dtype=np.int64)
    strided_conditional(out=out, N64=8 * step, S64=step)
    assert np.array_equal(out, np.arange(8, dtype=np.int64) * step), out


def test_power_bounds_declare_integer_parameters():
    """The frontend evaluates ``2**3`` to ``8.0``, as NumPy does; a map bound is an integer all the same."""

    @dace.program
    def powers(A: dace.float64[8], B: dace.float64[4]):
        for i in dace.map[0 : 2**3]:
            A[i] = 1
        for j in dace.map[0 : 2**40 : 2**38]:
            B[j // 2**38] = 1

    sdfg = powers.to_sdfg(simplify=True)
    assert all(h.startswith("for (int i = 0;") for h in _loop_headers(sdfg, "i")), _loop_headers(sdfg, "i")
    assert all(h.startswith("for (int64_t j = 0;") for h in _loop_headers(sdfg, "j")), _loop_headers(sdfg, "j")


def _consume_pe_header(num_pes: str, pe_type: dace.typeclass) -> str:
    sdfg = dace.SDFG("consume_pe_type")
    sdfg.add_symbol("P", pe_type)
    sdfg.add_stream("S", dace.int32, transient=True)
    sdfg.add_array("res", [1], dace.int32)
    state = sdfg.add_state()
    stream = state.add_access("S")
    output = state.add_write("res")
    entry, exit = state.add_consume("cons", ("p", num_pes))
    tasklet = state.add_tasklet("t", {"s"}, {"val"}, "val = s")
    state.add_edge(stream, None, entry, "IN_stream", dace.Memlet.from_array("S", sdfg.arrays["S"]))
    state.add_edge(entry, "OUT_stream", tasklet, "s", dace.Memlet.from_array("S", sdfg.arrays["S"]))
    state.add_edge(tasklet, "val", exit, "IN_V", dace.Memlet("res[0]", wcr="lambda a, b: a + b"))
    state.add_edge(exit, "OUT_V", output, None, dace.Memlet("res[0]", wcr="lambda a, b: a + b"))
    exit.add_in_connector("IN_V")
    exit.add_out_connector("OUT_V")
    code = sdfg.generate_code()[0].clean_code
    return next(line.strip() for line in code.splitlines() if "dace::Consume" in line)


def test_consume_pe_index_declares_inferred_type():
    assert "[&](int p," in _consume_pe_header("4", dace.int32)
    assert "[&](int64_t p," in _consume_pe_header("P", dace.int64)


if __name__ == "__main__":
    test_default_symbol_bound_declares_int()
    test_64bit_symbol_bound_declares_int64()
    test_out_of_range_literal_bound_declares_int64()
    test_negative_start_with_out_of_range_literal_stays_signed()
    test_integer_function_bound_keeps_argument_type()
    test_tiled_map_keeps_default_type()
    test_specialized_symbol_bound_keeps_its_type()
    test_dynamic_map_range_declares_connector_type()
    test_64bit_iteration_does_not_overflow()
    test_frontend_declares_map_parameter_with_its_range_type()
    test_frontend_map_parameter_beyond_32_bits()
    test_power_bounds_declare_integer_parameters()
    test_consume_pe_index_declares_inferred_type()
