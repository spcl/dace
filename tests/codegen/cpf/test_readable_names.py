# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A CPF unit names its loops ``i, j, k``, its temporaries without DaCe's underscores and its nested functions
briefly, keeps the names the program spelled, and computes what it computed before the renaming."""

import re

import numpy as np
import pytest

import dace
from dace.codegen.cpf import Rendering, render
from tests.codegen.cpf.conftest import assert_matches, assert_standalone, build_standalone, call_standalone

N = dace.symbol("N")

#: A generated identifier: one DaCe minted, which the rendering should not show. ``__x__`` spells a compiler keyword.
GENERATED = re.compile(r"(?<![\w.])_(?!_\w+__\b)\w+")


@dace.program
def names_axpy_and_scan(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    c[:] = a * 2.0 + b
    for k in range(1, N):
        a[k] = a[k - 1] + b[k]


@dace.program
def names_arguments_called_i_and_j(i: dace.float64[N], j: dace.float64[N]):
    j[:] = i + 1.0


def suffix(language: str) -> str:
    return "cpp" if language == "c++" else "c"


def generated_names(code: str) -> list[str]:
    body = code.split("DO NOT MODIFY", 1)[1]
    body = re.sub(r"#ifdef _OPENMP|#ifndef _OPENMP", "", body)
    return GENERATED.findall(body)


def function_text(code: str, name: str) -> str:
    start = re.search(rf"void {name}\(", code)
    assert start is not None, f"no function {name}"
    return code[start.start() : code.index("\n}\n", start.start())]


def run(result: Rendering, name: str, language: str, args: dict, expected: dict) -> None:
    assert_standalone(result.code, name, language=language)
    call_standalone(build_standalone(result.code, name, language=language), result.sdfg, args)
    assert_matches(expected, {key: args[key] for key in expected}, name)


@pytest.mark.parametrize("language", ["c++", "c"])
def test_array_expressions_render_with_letter_iterators_and_no_generated_names(language):
    name = f"cpf_names_axpy_{suffix(language)}"
    sdfg = names_axpy_and_scan.to_sdfg(simplify=True)
    sdfg.name = name

    result = render(sdfg, language=language)

    assert re.search(r"for \(int64_t i = 0;", result.code), result.code
    assert generated_names(result.code) == [], generated_names(result.code)
    rng = np.random.default_rng(0)
    a, b = rng.random(16), rng.random(16)
    expected_a = a.copy()
    for k in range(1, 16):
        expected_a[k] = expected_a[k - 1] + b[k]
    args = {"a": a.copy(), "b": b, "c": np.zeros(16), "N": 16}
    run(result, name, language, args, {"a": expected_a, "c": a * 2.0 + b})


@pytest.mark.parametrize("language", ["c++", "c"])
def test_an_iterator_skips_the_letters_the_arguments_hold(language):
    name = f"cpf_names_user_ij_{suffix(language)}"
    sdfg = names_arguments_called_i_and_j.to_sdfg(simplify=True)
    sdfg.name = name

    result = render(sdfg, language=language)

    assert re.search(r"for \(int64_t k = 0; k < N; k \+= 1\)", result.code), result.code
    assert not re.search(r"int64_t [ij]\b", result.code), result.code
    values = np.random.default_rng(1).random(9)
    args = {"i": values, "j": np.zeros(9), "N": 9}
    run(result, name, language, args, {"j": values + 1.0})


def team_sdfg(name: str) -> dace.SDFG:
    """An outer map over two parts calls a nested SDFG whose two maps cover the part's half of ``N`` with the same
    compound bounds, written over a symbol ``__part`` that the outer map parameter ``__t0`` binds."""
    inner = dace.SDFG("nested_sdfg")
    inner.add_symbol("__part", dace.int64)
    for array in ("a", "b", "c"):
        inner.add_array(array, [N], dace.float64)
    body = inner.add_state("body")
    half = "__part * int_ceil(N, 2):Min(N, (__part + 1) * int_ceil(N, 2))"
    body.add_mapped_tasklet(
        "double",
        {"__j0": half},
        {"x": dace.Memlet("a[__j0]")},
        "y = 2.0 * x",
        {"y": dace.Memlet("b[__j0]")},
        external_edges=True,
    )
    body.add_mapped_tasklet(
        "plus_one",
        {"__j1": half},
        {"x": dace.Memlet("a[__j1]")},
        "y = x + 1.0",
        {"y": dace.Memlet("c[__j1]")},
        external_edges=True,
    )

    sdfg = dace.SDFG(name)
    for array in ("a", "b", "c"):
        sdfg.add_array(array, [N], dace.float64)
    state = sdfg.add_state("main")
    node = state.add_nested_sdfg(inner, {"a": None}, {"b": None, "c": None}, {"__part": "__t0", "N": "N"})
    entry, exit_ = state.add_map("team", {"__t0": "0:2"})
    state.add_memlet_path(state.add_read("a"), entry, node, dst_conn="a", memlet=dace.Memlet("a[0:N]"))
    for array in ("b", "c"):
        state.add_memlet_path(node, exit_, state.add_write(array), src_conn=array, memlet=dace.Memlet(f"{array}[0:N]"))
    return sdfg


@pytest.mark.parametrize("language", ["c++", "c"])
def test_a_nested_function_reuses_its_callers_names_and_states_a_repeated_bound_once(language):
    name = f"cpf_names_team_{suffix(language)}"

    result = render(team_sdfg(name), language=language)

    body = function_text(result.code, "body")
    assert re.search(r"void body\([^)]*\bint64_t i\)", body), body
    assert body.count("lo = ") == 1 and body.count("hi = ") == 1, body
    # Every loop counter is unique in the file: the second sibling map numbers its counter
    assert body.count("for (int64_t j = lo; j < hi; j += 1)") == 1, body
    assert body.count("for (int64_t j2 = lo; j2 < hi; j2 += 1)") == 1, body
    assert re.search(r"for \(int64_t i = 0; i < 2; i \+= 1\)", result.code), result.code
    assert generated_names(result.code) == [], generated_names(result.code)
    a = np.random.default_rng(2).random(11)
    args = {"a": a, "b": np.zeros(11), "c": np.zeros(11), "N": 11}
    run(result, name, language, args, {"b": 2.0 * a, "c": a + 1.0})


def clashing_temporaries_sdfg(name: str) -> dace.SDFG:
    """``y = (x * 2 + tmp0) + 1`` through two generated temporaries whose plain names an argument and a C keyword
    already hold."""
    sdfg = dace.SDFG(name)
    for array in ("x", "tmp0", "y"):
        sdfg.add_array(array, [N], dace.float64)
    sdfg.add_transient("__tmp0", [N], dace.float64)
    sdfg.add_transient("__int", [N], dace.float64)
    state = sdfg.add_state()
    tmp, keyword = state.add_access("__tmp0"), state.add_access("__int")
    state.add_mapped_tasklet(
        "twice",
        {"__i0": "0:N"},
        {"u": dace.Memlet("x[__i0]"), "v": dace.Memlet("tmp0[__i0]")},
        "w = 2.0 * u + v",
        {"w": dace.Memlet("__tmp0[__i0]")},
        external_edges=True,
        output_nodes={"__tmp0": tmp},
    )
    state.add_mapped_tasklet(
        "plus_one",
        {"__i0": "0:N"},
        {"u": dace.Memlet("__tmp0[__i0]")},
        "w = u + 1.0",
        {"w": dace.Memlet("__int[__i0]")},
        external_edges=True,
        input_nodes={"__tmp0": tmp},
        output_nodes={"__int": keyword},
    )
    state.add_mapped_tasklet(
        "copy",
        {"__i0": "0:N"},
        {"u": dace.Memlet("__int[__i0]")},
        "w = u",
        {"w": dace.Memlet("y[__i0]")},
        external_edges=True,
        input_nodes={"__int": keyword},
    )
    return sdfg


@pytest.mark.parametrize("language", ["c++", "c"])
def test_a_temporary_named_like_an_argument_or_a_keyword_takes_a_numbered_name(language):
    name = f"cpf_names_clash_{suffix(language)}"

    result = render(clashing_temporaries_sdfg(name), language=language)

    assert re.search(r"\btmp0_1\[", result.code) and re.search(r"\bint1\[", result.code), result.code
    assert generated_names(result.code) == [], generated_names(result.code)
    x, extra = np.random.default_rng(3).random(7), np.random.default_rng(4).random(7)
    args = {"x": x, "tmp0": extra, "y": np.zeros(7), "N": 7}
    run(result, name, language, args, {"y": 2.0 * x + extra + 1.0})


@dace.program
def names_scatter(a: dace.float64[N], idx: dace.int64[N], b: dace.float64[N]):
    for i in range(N):
        b[idx[i]] = a[i] * 2.0


@pytest.mark.parametrize("language", ["c++", "c"])
def test_a_loop_comment_names_the_renamed_variables(language):
    name = f"cpf_names_hint_{suffix(language)}"
    sdfg = names_scatter.to_sdfg(simplify=True)
    sdfg.name = name

    result = render(sdfg, language=language)

    comments = [line for line in result.code.splitlines() if line.strip().startswith("//")]
    assert not [line for line in comments if GENERATED.search(line)], comments
    n = 12
    a, idx = np.random.default_rng(5).random(n), np.random.default_rng(6).permutation(n).astype(np.int64)
    expected = np.zeros(n)
    expected[idx] = 2.0 * a
    args = {"a": a, "idx": idx, "b": np.zeros(n), "N": n}
    run(result, name, language, args, {"b": expected})


M = dace.symbol("M")


@dace.program
def names_guarded_quotient_bounds(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    if M > 0:
        b[0 : N // M - 1] = a[0 : N // M - 1] * 2.0
        c[0 : N // M - 1] = a[0 : N // M - 1] + 1.0


@pytest.mark.parametrize("language", ["c++", "c"])
# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_a_repeated_bound_dividing_by_a_guarded_symbol_stays_in_its_branch(language):
    """``N // M - 1`` runs only under ``M > 0``; hoisted to the entry, ``M = 0`` divided by zero there."""
    name = f"cpf_names_guarded_bound_{suffix(language)}"
    sdfg = names_guarded_quotient_bounds.to_sdfg(simplify=True)
    sdfg.name = name
    result = render(sdfg, language=language)
    entry = result.sdfg.start_block
    hoisted = [e.data.assignments for e in result.sdfg.out_edges(entry) if entry.label.startswith("bounds")]
    assert not any("M" in value for assignments in hoisted for value in assignments.values()), hoisted
    library = build_standalone(result.code, name, language=language)
    for m in (0, 2):
        a = np.random.default_rng(m).random(12)
        b, c = np.zeros(12), np.zeros(12)
        args = {"a": a, "b": b.copy(), "c": c.copy(), "N": 12, "M": m}
        if m > 0:
            b[: 12 // m - 1], c[: 12 // m - 1] = a[: 12 // m - 1] * 2.0, a[: 12 // m - 1] + 1.0
        call_standalone(library, result.sdfg, args)
        assert_matches({"b": b, "c": c}, {"b": args["b"], "c": args["c"]}, name)


if __name__ == "__main__":
    for lang in ("c++", "c"):
        test_array_expressions_render_with_letter_iterators_and_no_generated_names(lang)
        test_an_iterator_skips_the_letters_the_arguments_hold(lang)
        test_a_nested_function_reuses_its_callers_names_and_states_a_repeated_bound_once(lang)
        test_a_temporary_named_like_an_argument_or_a_keyword_takes_a_numbered_name(lang)
        test_a_loop_comment_names_the_renamed_variables(lang)
        test_a_repeated_bound_dividing_by_a_guarded_symbol_stays_in_its_branch(lang)


def test_a_tasklets_file_scope_code_takes_the_renamed_iterator_without_a_shadow():
    """An external call's prototype names the loop iterator it receives (``tsvc_2_s231``): renaming the iterator to
    ``i`` renames it in the prototype too. A ``replace_dict`` shadow there (``auto _loop_it_1 = i;``) would sit at
    file scope, where ``i`` does not exist, and the unit would not compile."""
    from dace.sdfg.state import LoopRegion
    from dace.transformation.passes.readable_names import ReadableNames

    sdfg = dace.SDFG("call_per_iteration")
    sdfg.add_array("a", [10], dace.float64)
    loop = LoopRegion("steps", "_loop_it_1 < 10", "_loop_it_1", "_loop_it_1 = 0", "_loop_it_1 = _loop_it_1 + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    call = body.add_tasklet(
        "call",
        set(),
        {"out"},
        "call_one(&out, _loop_it_1);",
        language=dace.Language.CPP,
        code_global='extern "C" void call_one(double* out, int64_t _loop_it_1);',
    )
    body.add_edge(call, "out", body.add_write("a"), None, dace.Memlet("a[_loop_it_1]"))

    ReadableNames().apply_pass(sdfg, {})

    (renamed,) = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert loop.loop_variable == "i"
    assert renamed.code_global.as_string == 'extern "C" void call_one(double* out, int64_t i);'
    assert "auto" not in renamed.code.as_string and "_loop_it_1" not in renamed.code.as_string
