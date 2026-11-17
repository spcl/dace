# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A CPF unit says each thing once: what a loop class means sits in the header, a loop carries only its own
class and reason, and no line restates a DaCe-internal name. Each kernel is also built and run, so a shorter
text is never bought with a different result."""

import re

import numpy as np
import pytest

import dace
from dace import cpf_lowering
from dace.codegen.cpf import CONTRACT_LINES, EntrySignature, Rendering, render
from tests.codegen.cpf.conftest import assert_matches, assert_standalone, build_standalone, call_standalone

M = dace.symbol("M")
N = dace.symbol("N")


@dace.program
def cpf_compact_scatter(a: dace.float64[N], idx: dace.int64[N], b: dace.float64[N]):
    for i in range(N):
        b[idx[i]] = a[i] * 2.0
    for i in dace.map[0:N]:
        a[i] = a[i] + 1.0


def rendered(program, name: str, language: str, signature: EntrySignature | None = None) -> Rendering:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    return render(sdfg, language=language, signature=signature)


def run_scatter(result: Rendering, name: str, language: str) -> None:
    n = 40
    rng = np.random.default_rng(0)
    a = rng.random(n)
    idx = rng.permutation(n).astype(np.int64)
    expected_b = np.zeros(n)
    expected_b[idx] = a * 2.0
    args = {"a": a.copy(), "idx": idx, "b": np.zeros(n), "N": n}
    assert_standalone(result.code, name, language=language)
    call_standalone(build_standalone(result.code, name, language=language), result.sdfg, args)
    assert_matches({"a": a + 1.0, "b": expected_b}, {"a": args["a"], "b": args["b"]}, name)


def comment_lines(code: str) -> list[str]:
    return [line.strip() for line in code.splitlines() if line.strip().startswith("//")]


@pytest.mark.parametrize("language", ["c++", "c"])
def test_a_loop_comment_is_its_class_and_reason_and_the_header_says_what_the_class_means(language):
    name = f"cpf_compact_classes_{'cpp' if language == 'c++' else 'c'}"
    result = rendered(cpf_compact_scatter, name, language)
    comments = comment_lines(result.code)
    assert "// parallel -- the iterations are independent" in comments, comments
    assert any(re.match(r"// unknown -- .*\bb\b", line) for line in comments), comments
    body = result.code.split(CONTRACT_LINES[-1], 1)[1]
    for sentence in (cpf_lowering.SETTLED_PARALLEL, cpf_lowering.OPEN_UNKNOWN):
        assert result.code.count(sentence) == 1 and sentence not in body, sentence
    run_scatter(result, name, language)


def test_a_hint_stored_with_the_generic_sentence_renders_without_it():
    """An SDFG canonicalized before the sentences moved to the header carries them in its hints."""
    legacy = f"parallel -- the iterations are independent\n{cpf_lowering.SETTLED_PARALLEL}\nrun sequentially on CPU"
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE_C):
        comment = cpf_lowering.hint_comment(legacy)
    assert comment == "// parallel -- the iterations are independent\n// run sequentially on CPU\n"


@pytest.mark.parametrize("language", ["c++", "c"])
def test_no_statement_carries_its_tasklet_name(language):
    name = f"cpf_compact_labels_{'cpp' if language == 'c++' else 'c'}"
    result = rendered(cpf_compact_scatter, name, language)
    trailing = [line for line in result.code.splitlines() if re.search(r"[;{]\s*//", line)]
    assert not trailing, trailing
    run_scatter(result, name, language)


def broadcast_rows_sdfg(name: str, extent: str = "N") -> dace.SDFG:
    """``b[i, j] = a[i, j] + 1`` over an ``M x <extent>`` grid whose input repeats one row: ``a`` has row stride 0."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", ["M", extent], dace.float64, strides=[0, 1])
    sdfg.add_array("b", ["M", extent], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "plus_one",
        {"i": "0:M", "j": f"0:{extent}"},
        {"x": dace.Memlet("a[i, j]")},
        "y = x + 1.0",
        {"y": dace.Memlet("b[i, j]")},
        external_edges=True,
    )
    return sdfg


def index_helper(code: str, array: str) -> str:
    (helper,) = [line for line in code.splitlines() if re.search(rf"\binline int64_t {array}_idx\(", line)]
    return helper


def run_broadcast_rows(result: Rendering, name: str, language: str, extent: str = "N") -> None:
    m, n = 5, 7
    row = np.random.default_rng(1).random(n)
    args = {"a": row, "b": np.zeros((m, n)), "M": m, extent: n}
    call_standalone(build_standalone(result.code, name, language=language), result.sdfg, args)
    assert_matches({"b": np.tile(row + 1.0, (m, 1))}, {"b": args["b"]}, name)


@pytest.mark.parametrize("language", ["c++", "c"])
def test_an_index_helper_names_its_indices_i0_i1_on_one_line(language):
    name = f"cpf_compact_idx_names_{'cpp' if language == 'c++' else 'c'}"
    result = render(broadcast_rows_sdfg(name), language=language)
    assert re.search(r"\bb_idx\(int64_t i0, int64_t i1, int64_t N\) \{ return .*; \}$", index_helper(result.code, "b"))
    assert "__d" not in result.code
    run_broadcast_rows(result, name, language)


@pytest.mark.parametrize("language", ["c++", "c"])
def test_an_index_helper_takes_only_the_indices_its_body_reads(language):
    name = f"cpf_compact_idx_unused_{'cpp' if language == 'c++' else 'c'}"
    result = render(broadcast_rows_sdfg(name), language=language)
    helper = index_helper(result.code, "a")
    assert re.search(r"\ba_idx\(int64_t i1\) \{ return i1; \}$", helper), helper
    assert re.search(r"\ba\[a_idx\(\w+\)\]", result.code), result.code
    run_broadcast_rows(result, name, language)


def test_an_extent_named_like_an_index_keeps_the_helper_indices_apart():
    name = "cpf_compact_idx_clash"
    result = render(broadcast_rows_sdfg(name, extent="i1"), language="c")
    assert re.search(r"\bb_idx\(int64_t __d0, int64_t __d1, int64_t i1\)", index_helper(result.code, "b"))
    run_broadcast_rows(result, name, "c", extent="i1")


if __name__ == "__main__":
    test_a_loop_comment_is_its_class_and_reason_and_the_header_says_what_the_class_means("c++")
    test_a_loop_comment_is_its_class_and_reason_and_the_header_says_what_the_class_means("c")
    test_a_hint_stored_with_the_generic_sentence_renders_without_it()
    test_no_statement_carries_its_tasklet_name("c++")
    test_no_statement_carries_its_tasklet_name("c")
    test_an_index_helper_names_its_indices_i0_i1_on_one_line("c++")
    test_an_index_helper_names_its_indices_i0_i1_on_one_line("c")
    test_an_index_helper_takes_only_the_indices_its_body_reads("c++")
    test_an_index_helper_takes_only_the_indices_its_body_reads("c")
    test_an_extent_named_like_an_index_keeps_the_helper_indices_apart()
