# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A CPF unit carries none of the generator's own spellings: no connector temporaries, no ``__acc_`` or
``__g_`` names, no ``*(&x)`` or ``*(p + e)`` where the reads write ``x`` and ``p[e]``."""

import math
import re

import numpy as np

import dace
from dace.codegen import cpf_compact
from dace.codegen.cpf import collapse_connector_blocks, render
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def gather_count(a: dace.float64[N], idx: dace.int64[N], b: dace.float64[N], counts: dace.int64[N]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            arr << a[:]
            j << idx[i]
            out >> b[i]
            out = arr[j]
        counts[idx[i]] += 1


@dace.program
def row_norms(m: dace.float64[N, 3], out: dace.float64[N]):
    for i in dace.map[0:N]:
        s = 0.0
        for j in dace.map[0:3]:
            s += m[i, j] * m[i, j]
        out[i] += s


@dace.program
def row_sums(m: dace.float64[N, 3], out: dace.float64[N, 2]):
    for i in dace.map[0:N]:
        for j in dace.map[0:3]:
            out[i, 1] += m[i, j]


@dace.program
def scatter_2d(idx: dace.int64[N], counts: dace.int64[N, 2]):
    for i in dace.map[0:N]:
        counts[idx[i], 1] += 1


@dace.program
def sign_max(x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << x[i]
            b >> y[i]
            b = max(math.copysign(1.0, a), 0.5 * a)


TABLE = np.array([0.5, 1.5, 2.5, 3.5])


def two_state_nest(name: str, constant_input: bool, op: str) -> dace.SDFG:
    """``y[i] = x[i] <op> c``, where ``c`` is ``table[idx[i]]`` (the file-scope table) or the scalar input ``eps``. Its
    two states keep it a helper inside the map that calls it."""
    nest = dace.SDFG(name)
    for array, dtype in (("x", dace.float64), ("idx", dace.int64), ("y", dace.float64)):
        nest.add_array(array, [N], dtype)
    nest.add_scalar("j", dace.int64, transient=True)
    nest.add_symbol("i", dace.int64)
    first = nest.add_state("first")
    pick = first.add_tasklet("pick", {"k"}, {"o"}, "o = k")
    first.add_edge(first.add_read("idx"), None, pick, "k", dace.Memlet("idx[i]"))
    first.add_edge(pick, "o", first.add_write("j"), None, dace.Memlet("j[0]"))
    second = nest.add_state_after(first, "second")
    if constant_input:
        nest.add_scalar("eps", dace.float64)
        scale = second.add_tasklet("scale", {"v", "e", "k"}, {"o"}, "o = v * e + k")
        second.add_edge(second.add_read("eps"), None, scale, "e", dace.Memlet("eps[0]"))
    else:
        nest.add_constant("table", TABLE)
        scale = second.add_tasklet("scale", {"v", "k"}, {"o"}, f"o = v {op} table[k]")
    second.add_edge(second.add_read("x"), None, scale, "v", dace.Memlet("x[i]"))
    second.add_edge(second.add_read("j"), None, scale, "k", dace.Memlet("j[0]"))
    second.add_edge(scale, "o", second.add_write("y"), None, dace.Memlet("y[i]"))
    return nest


def calling_nests(name: str, outputs: tuple[str, ...], constant_input: bool) -> dace.SDFG:
    """One map per output, each calling its own :func:`two_state_nest` (``*`` then ``+``); the root holds ``table`` and ``eps``."""
    sdfg = dace.SDFG(name)
    for array, dtype in (("x", dace.float64), ("idx", dace.int64), *((out, dace.float64) for out in outputs)):
        sdfg.add_array(array, [N], dtype)
    sdfg.add_constant("table", TABLE)
    sdfg.add_scalar("eps", dace.float64, transient=True)
    sdfg.add_constant("eps", 0.5, sdfg.arrays["eps"])
    state = None
    for out, op in zip(outputs, "*+"):
        state = sdfg.add_state(f"fill_{out}") if state is None else sdfg.add_state_after(state, f"fill_{out}")
        inputs = {"x", "idx", "eps"} if constant_input else {"x", "idx"}
        node = state.add_nested_sdfg(
            two_state_nest(f"fill_{out}", constant_input, op), inputs, {"y"}, {"i": "i", "N": "N"}
        )
        entry, exit_ = state.add_map(f"rows_{out}", {"i": "0:N"})
        for data in sorted(inputs):
            subset = "eps[0]" if data == "eps" else f"{data}[0:N]"
            state.add_memlet_path(state.add_read(data), entry, node, dst_conn=data, memlet=dace.Memlet(subset))
        state.add_memlet_path(node, exit_, state.add_write(out), src_conn="y", memlet=dace.Memlet(f"{out}[0:N]"))
    return sdfg


def nest_inputs():
    rng = np.random.default_rng(5)
    n = 17
    return n, rng.random(n), rng.integers(0, len(TABLE), n)


def connector_names(code: str) -> list[str]:
    return re.findall(r"\b__(?:in|out|arr)\w*", code)


def test_connector_temporaries_of_any_type_fold_and_compute_the_same():
    sdfg = gather_count.to_sdfg(simplify=True)
    sdfg.name = "clutter_gather_count"
    code = render(sdfg, language="c").code
    assert not connector_names(code), code
    rng = np.random.default_rng(11)
    n = 23
    a, idx = rng.random(n), rng.integers(0, 5, n)
    b, counts = np.zeros(n), np.zeros(n, dtype=np.int64)
    args = {"a": a, "idx": idx, "b": b, "counts": counts, "N": n}
    call_standalone(build_standalone(code, "clutter_gather_count", language="c"), sdfg, args)
    np.testing.assert_array_equal(b, a[idx])
    np.testing.assert_array_equal(counts, np.bincount(idx, minlength=n))


def test_a_connector_local_of_another_type_keeps_its_conversion():
    types = {"n": "int64_t", "m": "int64_t", "x": "double"}
    divide = "    {\n        double __in2 = n;\n        alpha = (m / __in2);\n    }\n"
    assert collapse_connector_blocks(divide, types) == divide, "folding would make the division an integer one"
    both = "    {\n        int64_t __in1 = m;\n        int64_t __in2 = n;\n        k = (__in1 * __in2);\n    }\n"
    assert collapse_connector_blocks(both, types) == "    k = (m * n);\n"
    view = "    {\n        const double* restrict __arr = &x[0];\n        t = __arr[n];\n    }\n"
    assert collapse_connector_blocks(view, types) == "    t = x[n];\n"
    types |= {"f": "bool", "p": "bool", "q": "bool"}
    logical = "    {\n        bool __out;\n        __out = (p[i] == q[i]);\n        f[0] = f[0] && (__out);\n    }\n"
    assert collapse_connector_blocks(logical, types) == "    f[0] = f[0] && (p[i] == q[i]);\n"
    unread = "    {\n        double __out;\n        __out = (x[0] + x[1]);\n    }\n"
    assert collapse_connector_blocks(unread, types) == "", "an unread pure value computes nothing"


def test_accumulators_and_constants_get_short_names():
    sdfg = row_sums.to_sdfg(simplify=True)
    sdfg.name = "clutter_row_sums"
    code = render(sdfg, language="c", llm_friendly=True).code
    assert "__acc_" not in code and re.search(r"\bout_acc0\b", code), code
    rng = np.random.default_rng(13)
    m, out = rng.random((9, 3)), rng.random((9, 2))
    want = out.copy()
    want[:, 1] += m.sum(axis=1)
    call_standalone(build_standalone(code, "clutter_row_sums", language="c"), sdfg, {"m": m, "out": out, "N": 9})
    np.testing.assert_allclose(out, want)
    constant = dace.SDFG("clutter_constant")
    constant.add_constant("__g_numpy_arange_3", np.arange(3, dtype=np.int64))
    unit = "constexpr int64_t __g_numpy_arange_3[3] = {0, 1, 2};\nvoid clutter_constant(int64_t* o)\n{\n"
    unit += "    o[0] = __g_numpy_arange_3[2];\n}\n"
    assert "__g_" not in cpf_compact.compact(unit, constant, {"clutter_constant", "o"}).code


def test_a_local_scalar_accumulates_in_place():
    sdfg = row_norms.to_sdfg(simplify=True)
    sdfg.name = "clutter_row_norms"
    code = render(sdfg, language="c").code
    assert "__acc_" not in code and re.search(r"\bs = s \+ ", code), code
    rng = np.random.default_rng(17)
    m, out = rng.random((9, 3)), rng.random(9)
    want = out + (m * m).sum(axis=1)
    call_standalone(build_standalone(code, "clutter_row_norms", language="c"), sdfg, {"m": m, "out": out, "N": 9})
    np.testing.assert_allclose(out, want)


def test_reduction_writes_index_as_reads_do():
    sdfg = row_sums.to_sdfg(simplify=True)
    sdfg.name = "clutter_row_sums_idx"
    code = render(sdfg, language="c").code
    assert re.search(r"\bout\[out_idx\(i, 1\)\]", code) and "*(" not in code.split("clutter_row_sums_idx(")[-1], code
    scatter = scatter_2d.to_sdfg(simplify=True)
    scatter.name = "clutter_scatter_2d"
    code = render(scatter, language="c").code
    assert re.search(r"#pragma omp atomic update\n\s*counts\[counts_idx\(", code), code
    rng = np.random.default_rng(19)
    idx, counts = rng.integers(0, 6, 11), np.zeros((11, 2), dtype=np.int64)
    call_standalone(
        build_standalone(code, "clutter_scatter_2d", language="c"), scatter, {"idx": idx, "counts": counts, "N": 11}
    )
    np.testing.assert_array_equal(counts[:, 1], np.bincount(idx, minlength=11))


def test_copysign_inside_a_max_renders_in_c():
    sdfg = sign_max.to_sdfg(simplify=True)
    sdfg.name = "clutter_sign_max"
    code = render(sdfg, language="c").code
    x, y = np.linspace(-2.0, 2.0, 7), np.zeros(7)
    call_standalone(build_standalone(code, "clutter_sign_max", language="c"), sdfg, {"x": x, "y": y, "N": 7})
    np.testing.assert_array_equal(y, np.maximum(np.copysign(1.0, x), 0.5 * x))


def test_pointer_writes_are_spelled_as_subscripts():
    sdfg = row_norms.to_sdfg(simplify=True)
    sdfg.name = "clutter_subscripts"
    code = render(sdfg, language="c").code
    assert not re.search(r"\*\(&\w+\)|\*\(\w+ \+ ", code), code
    assert re.search(r"\bout\[i\] = out\[i\] \+ s;", code), code


def test_helpers_share_one_file_scope_table():
    sdfg = calling_nests("clutter_shared_table", ("y", "z"), constant_input=False)
    code = render(sdfg, language="c").code
    assert len(re.findall(r"constexpr double table\[4\]", code)) == 1, code
    assert len(re.findall(r"static inline void \w+\(", code)) == 2, code
    n, x, idx = nest_inputs()
    y, z = np.zeros(n), np.zeros(n)
    args = {"x": x, "idx": idx, "y": y, "z": z, "N": n}
    call_standalone(build_standalone(code, "clutter_shared_table", language="c"), sdfg, args)
    np.testing.assert_array_equal(y, x * TABLE[idx])
    np.testing.assert_array_equal(z, x + TABLE[idx])


def test_a_helper_reads_a_file_scope_constant_by_name():
    sdfg = calling_nests("clutter_shared_scalar", ("y",), constant_input=True)
    code = render(sdfg, language="c").code
    helpers = re.findall(r"static inline void \w+\(([^)]*)\)", code)
    assert len(helpers) == 1 and not re.search(r"\beps\b", helpers[0]), code
    assert len(re.findall(r"constexpr double eps\b", code)) == 1 and not connector_names(code), code
    n, x, idx = nest_inputs()
    y = np.zeros(n)
    call_standalone(
        build_standalone(code, "clutter_shared_scalar", language="c"), sdfg, {"x": x, "idx": idx, "y": y, "N": n}
    )
    np.testing.assert_array_equal(y, x * 0.5 + idx)


if __name__ == "__main__":
    test_connector_temporaries_of_any_type_fold_and_compute_the_same()
    test_a_connector_local_of_another_type_keeps_its_conversion()
    test_accumulators_and_constants_get_short_names()
    test_a_local_scalar_accumulates_in_place()
    test_reduction_writes_index_as_reads_do()
    test_copysign_inside_a_max_renders_in_c()
    test_pointer_writes_are_spelled_as_subscripts()
    test_helpers_share_one_file_scope_table()
    test_a_helper_reads_a_file_scope_constant_by_name()
