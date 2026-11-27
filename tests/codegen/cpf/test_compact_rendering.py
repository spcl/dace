# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A CPF unit says each thing once: what a loop class means sits in the header, a loop carries only its own
class and reason, and no line restates a DaCe-internal name. Each kernel is also built and run, so a shorter
text is never bought with a different result."""

import re

import numpy as np
import pytest

import dace
from dace import cpf_lowering
from dace.codegen.cpf import CONTRACT_LINES, EntrySignature, Rendering, compact_blocks, render
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.annotate_loop_kinds import PARALLEL
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
    assert f"// {PARALLEL}" in comments, comments
    assert any(re.match(r"// unknown -- .*\bb\b.*; parallel if ", line) for line in comments), comments
    # What each class means is said once, in the header, and never under a loop
    body = result.code.split(CONTRACT_LINES[-1], 1)[1]
    assert all(line not in body for line in CONTRACT_LINES)
    for sentence in (cpf_lowering.SETTLED_PARALLEL, cpf_lowering.OPEN_UNKNOWN):
        assert sentence not in result.code, sentence
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


#: Bytes past the end of the workspace the tests fill and check: a slot past the declared size lands there.
CANARY = 256
WORKSPACE = EntrySignature(("a", "out", "N", "workspace", "workspace_size"), "workspace", "workspace_size")


def loop_scratch_sdfg(name: str, zeroed: bool) -> dace.SDFG:
    """Three times over: ``tmp = a + t`` and ``out += reverse(tmp)``, or with ``zeroed`` ``tmp += a`` on a ``tmp``
    zeroed each time it is allocated and ``out += tmp``. ``tmp`` and ``tmp2`` are allocated in the loop body."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    for scratch in ("tmp", "tmp2"):
        sdfg.add_transient(scratch, [N], dace.float64, lifetime=dace.AllocationLifetime.State)
    loop = LoopRegion("repeat", condition_expr="t < 3", loop_var="t", initialize_expr="t = 0", update_expr="t = t + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tmp = body.add_access("tmp")
    tmp.setzero = zeroed
    tmp2 = body.add_access("tmp2")
    if zeroed:
        body.add_mapped_tasklet(
            "acc",
            {"i": "0:N"},
            {"x": dace.Memlet("a[i]")},
            "y = x",
            {"y": dace.Memlet("tmp[i]", wcr="lambda p, q: p + q")},
            external_edges=True,
            output_nodes={"tmp": tmp},
        )
        source = "tmp2[i]"
    else:
        body.add_mapped_tasklet(
            "fill",
            {"i": "0:N"},
            {"x": dace.Memlet("a[i]")},
            "y = x + t",
            {"y": dace.Memlet("tmp[i]")},
            external_edges=True,
            output_nodes={"tmp": tmp},
        )
        source = "tmp2[N - 1 - i]"
    body.add_mapped_tasklet(
        "copy",
        {"i": "0:N"},
        {"x": dace.Memlet("tmp[i]")},
        "y = x",
        {"y": dace.Memlet("tmp2[i]")},
        external_edges=True,
        input_nodes={"tmp": tmp},
        output_nodes={"tmp2": tmp2},
    )
    body.add_mapped_tasklet(
        "add",
        {"i": "0:N"},
        {"x": dace.Memlet(source), "o": dace.Memlet("out[i]")},
        "y = o + x",
        {"y": dace.Memlet("out[i]")},
        external_edges=True,
        input_nodes={"tmp2": tmp2},
    )
    return sdfg


def run_with_canary(result: Rendering, name: str, language: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(a, out)`` after one call on a workspace of exactly ``result.workspace_bytes`` bytes, checking that the
    bytes past it are untouched."""
    needed = int(eval(result.workspace_bytes, {"N": n}))  # noqa: S307 -- an expression the renderer wrote
    raw = np.zeros(needed + CANARY + 64, dtype=np.uint8)
    start = (-raw.ctypes.data) % 64
    workspace = raw[start : start + needed + CANARY]
    workspace[needed:] = 0xA5
    a = np.random.default_rng(2).random(n)
    args = {"a": a, "out": np.zeros(n), "N": n, "workspace": workspace, "workspace_size": needed}
    library = build_standalone(result.code, name, language=language)
    call_standalone(library, result.sdfg, args, order=result.arguments)
    assert (workspace[needed:] == 0xA5).all(), "a workspace slot reached past the declared size"
    return a, args["out"]


@pytest.mark.parametrize("language", ["c++", "c"])
def test_the_workspace_is_carved_at_fixed_offsets_taken_once_per_call(language):
    """``tmp`` and ``tmp2`` are allocated in the loop body, yet each points at its slot once, at the top of the entry:
    a slot taken per iteration would reach past the declared size."""
    name = f"cpf_compact_carve_{'cpp' if language == 'c++' else 'c'}"
    result = render(loop_scratch_sdfg(name, zeroed=False), language=language, signature=WORKSPACE)
    code = result.code
    cast = r"\(double \*\)\(" if language == "c" else r"reinterpret_cast<double \*>\("
    for scratch in ("tmp", "tmp2"):
        slots = re.findall(rf"^\s*{scratch} = {cast}workspace \+ .*\);$", code, re.MULTILINE)
        assert len(slots) == 1 and code.index(slots[0].strip()) < code.index("for (int64_t t ="), code
    a, out = run_with_canary(result, name, language, n=33)
    assert_matches({"out": sum(a[::-1] + t for t in range(3))}, {"out": out}, name)


def test_an_array_zeroed_on_each_allocation_keeps_its_own_allocation():
    """Allocated once at the top it would be zeroed once, and the loop would accumulate into stale values."""
    name = "cpf_compact_carve_zeroed"
    result = render(loop_scratch_sdfg(name, zeroed=True), language="c", signature=WORKSPACE)
    assert re.search(r"\btmp2 = \(double \*\)\(workspace \+ workspace_off\)", result.code), result.code
    assert not re.search(r"\btmp = \(double \*\)\(workspace", result.code), result.code
    a, out = run_with_canary(result, name, "c", n=33)
    assert_matches({"out": 3 * a}, {"out": out}, name)


GROUPING_ONLY = """void f(double *a, int n)
{
    a[0] = 1.0;
    {

        a[1] = 2.0;
        {
            if (n > 2) {
                a[2] = 3.0;
            }
        }

    }
}"""


def test_a_block_that_only_groups_loses_its_braces_and_blank_lines():
    assert compact_blocks(GROUPING_ONLY) == (
        "void f(double *a, int n)\n{\n    a[0] = 1.0;\n    a[1] = 2.0;\n    if (n > 2) {\n        a[2] = 3.0;\n    }\n}"
    )


KEPT_BLOCKS = {
    "a declaration": "x = 1;\n{\n    double t;\n    t = 2.0;\n}",
    "a qualified declaration": "x = 1;\n{\n    const int64_t n = 4;\n}",
    "a pointer declaration": "x = 1;\n{\n    double* restrict p = q;\n}",
    "a for body": "for (i = 0; i < n; i += 1)\n{\n    x = i;\n}",
    "an else body": "if (c) {\n    x = 1;\n} else\n{\n    x = 2;\n}",
    "an OpenMP structured block": "x = 1;\n#pragma omp parallel\n{\n    f();\n}",
    "a trailing directive": "x = 1;\n{\n    f();\n    #pragma omp barrier\n}",
    "a label": "x = 1;\n{\n    done:\n    x = 2;\n}",
    "a wrapped statement": "x = 1;\n{\n    f(a,\n      b);\n}",
}


@pytest.mark.parametrize("text", list(KEPT_BLOCKS.values()), ids=list(KEPT_BLOCKS))
def test_a_block_that_scopes_or_is_a_body_keeps_its_braces(text):
    code = "void f(void)\n{\n" + text + "\n}"
    assert compact_blocks(code) == code


def test_braces_in_comments_and_literals_do_not_pair():
    code = "void f(void)\n{\n    x = 1;\n    {\n        s = \"}\"; // {\n        c = '{'; /* } */\n    }\n}"
    assert compact_blocks(code) == "void f(void)\n{\n    x = 1;\n    s = \"}\"; // {\n    c = '{'; /* } */\n}"


def test_blank_lines_between_definitions_collapse_to_one():
    assert compact_blocks("int a;\n\n\n\nint b;\n") == "int a;\n\nint b;\n"


@dace.program
def cpf_compact_branches(a: dace.float64[N], b: dace.float64[N], flag: dace.int64):
    if flag > 0:
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0
    else:
        for i in dace.map[0:N]:
            b[i] = a[i] + 3.0


@pytest.mark.parametrize("language", ["c++", "c"])
def test_a_compacted_unit_has_no_blank_line_in_a_body_and_computes_the_same(language):
    name = f"cpf_compact_blocks_{'cpp' if language == 'c++' else 'c'}"
    result = rendered(cpf_compact_branches, name, language)
    entry = result.code[result.code.index(f"{name}(") :]
    assert not [line for line in entry.splitlines() if not line.strip()], entry
    assert_standalone(result.code, name, language=language)
    library = build_standalone(result.code, name, language=language)
    a = np.random.default_rng(3).random(17)
    for flag, expected in ((1, a * 2.0), (0, a + 3.0)):
        args = {"a": a, "b": np.zeros(17), "N": 17, "flag": flag}
        call_standalone(library, result.sdfg, args)
        assert_matches({"b": expected}, {"b": args["b"]}, name)


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
    test_the_workspace_is_carved_at_fixed_offsets_taken_once_per_call("c++")
    test_the_workspace_is_carved_at_fixed_offsets_taken_once_per_call("c")
    test_an_array_zeroed_on_each_allocation_keeps_its_own_allocation()
    test_a_block_that_only_groups_loses_its_braces_and_blank_lines()
    for text in KEPT_BLOCKS.values():
        test_a_block_that_scopes_or_is_a_body_keeps_its_braces(text)
    test_braces_in_comments_and_literals_do_not_pair()
    test_blank_lines_between_definitions_collapse_to_one()
    test_a_compacted_unit_has_no_blank_line_in_a_body_and_computes_the_same("c++")
    test_a_compacted_unit_has_no_blank_line_in_a_body_and_computes_the_same("c")
