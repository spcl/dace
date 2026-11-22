# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A CPF unit carries no text an agent reads for nothing: a scalar one tasklet writes and one other reads is folded
into its reader, and a 1-D unit-stride array is indexed directly instead of through an identity ``<array>_idx``."""

import math
import re

import numpy as np

import dace
from dace.codegen.cpf import collapse_connector_blocks, render
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def stencil_1d(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[1 : N - 1]:
        tmp = dace.define_local_scalar(dace.float64)
        with dace.tasklet:
            left << a[i - 1]
            mid << a[i]
            right << a[i + 1]
            curvature >> tmp
            curvature = left - 2.0 * mid + right
        with dace.tasklet:
            c << tmp
            m << a[i]
            out >> b[i]
            out = 0.25 * c + m


def test_single_use_temporaries_fold_into_their_reader_and_compute_the_same():
    sdfg = stencil_1d.to_sdfg(simplify=True)
    sdfg.name = "token_cuts_stencil"
    code = render(sdfg, language="c").code
    body = code[code.index("token_cuts_stencil(") :]
    assert not re.search(r"^\s*double \w+;$", body, re.MULTILINE), body
    rng = np.random.default_rng(3)
    for n in (3, 17):
        a, b = rng.random(n), np.zeros(n)
        call_standalone(build_standalone(code, "token_cuts_stencil", language="c"), sdfg, {"a": a, "b": b, "N": n})
        want = np.zeros(n)
        want[1:-1] = 0.25 * (a[:-2] - 2.0 * a[1:-1] + a[2:]) + a[1:-1]
        np.testing.assert_array_equal(b[1:-1], want[1:-1])


def test_a_unit_stride_vector_is_indexed_without_an_identity_helper():
    sdfg = stencil_1d.to_sdfg(simplify=True)
    sdfg.name = "token_cuts_identity"
    code = render(sdfg, language="c").code
    assert not re.search(r"_idx\(int64_t i0\) \{ return i0; \}", code), code
    assert re.search(r"\ba\[\(?i", code), code


@dace.program
def one_trip(a: dace.float64[N], s: dace.float64[1]):
    for k in dace.map[0:1]:
        s[k] = a[0] * 2.0


@dace.program
def team_one_trip(a: dace.float64[N], s: dace.float64[1]):
    for tid in dace.map[0:1] @ dace.ScheduleType.CPU_Persistent:
        for k in dace.map[0:1] @ dace.ScheduleType.CPU_Multicore:
            s[k] = a[0] + tid


def one_trip_maps(sdfg: dace.SDFG) -> int:
    return sum(
        1
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry) and all(b == e for b, e, step in node.map.range)
    )


def test_a_one_trip_loop_outside_a_team_becomes_straight_line_code():
    sdfg = one_trip.to_sdfg(simplify=True)
    sdfg.name = "token_cuts_one_trip"
    assert one_trip_maps(sdfg) == 1, "the program must reach CPF with its one-trip map"
    code = render(sdfg, language="c").code
    assert not re.search(r"for \(int64_t \w+ = 0; \w+ < 1;", code), code
    a, s = np.arange(1.0, 5.0), np.zeros(1)
    call_standalone(build_standalone(code, "token_cuts_one_trip", language="c"), sdfg, {"a": a, "s": s, "N": 4})
    assert s[0] == 2.0


def test_inside_a_team_a_one_trip_loop_becomes_a_single_block_and_the_team_stays():
    sdfg = team_one_trip.to_sdfg(simplify=True)
    sdfg.name = "token_cuts_team_one_trip"
    assert one_trip_maps(sdfg) == 2, "the program must reach CPF with its team and its one-trip map"
    code = render(sdfg, language="c").code
    assert "#pragma omp parallel" in code and "#pragma omp single" in code, code
    assert "#pragma omp for" not in code and not re.search(r"\w+ < 1;", code), code
    a, s = np.arange(1.0, 5.0), np.zeros(1)
    call_standalone(build_standalone(code, "token_cuts_team_one_trip", language="c"), sdfg, {"a": a, "s": s, "N": 4})
    assert s[0] == 1.0


def test_a_connector_block_becomes_its_one_statement():
    wcr = "    {\n        double __out;\n        __out = (x[i] * y[i]);\n        *(acc) = *(acc) + (__out);\n    }\n"
    assert collapse_connector_blocks(wcr) == "    *(acc) = *(acc) + (x[i] * y[i]);\n"
    read = "    {\n        double __inp = acc[0];\n        a[i] = __inp;\n    }\n"
    assert collapse_connector_blocks(read) == "    a[i] = acc[0];\n"
    twice = "    {\n        double __out;\n        __out = f(i);\n        s = __out * __out;\n    }\n"
    assert collapse_connector_blocks(twice) == twice, "a value used twice must be evaluated once"
    narrow = "    {\n        float __out;\n        __out = (x[i] * 2.0);\n        *(acc) = *(acc) + (__out);\n    }\n"
    assert collapse_connector_blocks(narrow) == narrow, "a float local rounds its value: folding it in changes the sum"
    named = "    {\n        double y;\n        y = (x[i] * 2.0);\n        *(acc) = *(acc) + (y);\n    }\n"
    assert collapse_connector_blocks(named) == named, "a tasklet's own local keeps its declaration"
    atomic = (
        "    {\n        double __out;\n        __out = (x[i] * y[i]);\n        // atomic update: why\n"
        "        #pragma omp atomic update\n        *(s) = *(s) + (__out);\n    }\n"
    )
    assert collapse_connector_blocks(atomic) == (
        "    // atomic update: why\n    #pragma omp atomic update\n    *(s) = *(s) + (x[i] * y[i]);\n"
    )


@dace.program
def dot(x: dace.float64[N], y: dace.float64[N], s: dace.float64[1]):
    for tid in dace.map[0:1] @ dace.ScheduleType.CPU_Persistent:
        for i in dace.map[0:N] @ dace.ScheduleType.CPU_Multicore:
            s[0] += x[i] * y[i]


def test_a_reduction_update_is_one_statement_and_sums_the_same():
    sdfg = dot.to_sdfg(simplify=True)
    sdfg.name = "token_cuts_dot"
    code = render(sdfg, language="c").code
    assert "double __out;" not in code, code
    rng = np.random.default_rng(5)
    x, y, s = rng.random(33), rng.random(33), np.zeros(1)
    call_standalone(build_standalone(code, "token_cuts_dot", language="c"), sdfg, {"x": x, "y": y, "s": s, "N": 33})
    np.testing.assert_allclose(s[0], x @ y)


M = dace.symbol("M", dtype=dace.int64)


@dace.program
def fp32_chain(a: dace.float32[M], b: dace.float32[M]):
    for i in dace.map[0:M]:
        tmp = dace.define_local_scalar(dace.float32)
        with dace.tasklet:
            x << a[i]
            t >> tmp
            t = x * 2.0
        with dace.tasklet:
            u << tmp
            out >> b[i]
            out = u + 1.0


def test_a_float_temporary_keeps_its_rounding():
    """An fp32 intermediate rounds the double ``x * 2.0``; folded into ``out`` the sum would round once, not twice."""
    sdfg = fp32_chain.to_sdfg(simplify=True)
    sdfg.name = "token_cuts_fp32"
    code = render(sdfg, language="c").code
    assert re.search(r"^\s*float \w+;$", code, re.MULTILINE), code


@dace.program
def gather_exp(a: dace.float64[N], idx: dace.int64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        tmp = dace.define_local_scalar(dace.float64)
        with dace.tasklet:
            arr << a[:]
            j << idx[i]
            t >> tmp
            t = -arr[j]
        with dace.tasklet:
            u << tmp
            out >> b[i]
            out = math.exp(u)


def test_an_indirect_read_is_not_folded_into_a_maths_call():
    """The C printer cannot type ``exp(-__arr[j])``: it would not know to spell ``exp`` (lavamd, ls3df_scf)."""
    sdfg = gather_exp.to_sdfg(simplify=True)
    sdfg.name = "token_cuts_gather_exp"
    code = render(sdfg, language="c").code
    rng = np.random.default_rng(7)
    a, idx, b = rng.random(9), rng.integers(0, 9, 9), np.zeros(9)
    call_standalone(
        build_standalone(code, "token_cuts_gather_exp", language="c"), sdfg, {"a": a, "idx": idx, "b": b, "N": 9}
    )
    np.testing.assert_allclose(b, np.exp(-a[idx]))


if __name__ == "__main__":
    test_single_use_temporaries_fold_into_their_reader_and_compute_the_same()
    test_a_unit_stride_vector_is_indexed_without_an_identity_helper()
    test_a_one_trip_loop_outside_a_team_becomes_straight_line_code()
    test_inside_a_team_a_one_trip_loop_becomes_a_single_block_and_the_team_stays()
    test_a_connector_block_becomes_its_one_statement()
    test_a_reduction_update_is_one_statement_and_sums_the_same()
    test_a_float_temporary_keeps_its_rounding()
    test_an_indirect_read_is_not_folded_into_a_maths_call()
