# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The LLM-friendly spelling of a CPF unit computes and exports what the rendering does, under short names it can
revert; anything it cannot respell safely it leaves alone."""

import re

import numpy as np
import pytest

import dace
from dace.codegen.cpf import Rendering, render
from dace.codegen.cpf_compact import Compacted, compact, drop_unused, exported_names, respell, tokens
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def staged_stencil(a: dace.float64[N], b: dace.float64[N], steps: dace.int64):
    for step in range(steps):
        tmp_buffer_for_the_stencil = np.zeros_like(a)
        tmp_buffer_for_the_stencil[1:-1] = 0.25 * (a[:-2] + 2.0 * a[1:-1] + a[2:])
        a[1:-1] = tmp_buffer_for_the_stencil[1:-1]
    b[:] = np.cumsum(a)


def identifiers(code: str) -> set[str]:
    return {piece for kind, piece in tokens(code) if kind == "ident"}


def run(rendering: Rendering, name: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    a, b = rng.random(n), np.zeros(n)
    arguments = {"a": a, "b": b, "N": n, "steps": 3}
    call_standalone(build_standalone(rendering.code, name, language="c"), rendering.sdfg, arguments)
    return a, b


def renderings(name: str) -> tuple[Rendering, Rendering]:
    sdfg = staged_stencil.to_sdfg(simplify=True)
    sdfg.name = name
    return render(sdfg, language="c"), render(sdfg, language="c", llm_friendly=True)


def test_the_llm_friendly_unit_computes_the_same_values_in_fewer_characters():
    plain, friendly = renderings("llm_friendly_stencil")
    assert len(friendly.code) < len(plain.code)
    for n in (1, 7, 64):
        want = run(plain, "llm_friendly_stencil_ref", n)
        got = run(friendly, "llm_friendly_stencil_compact", n)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])


def test_generated_names_shorten_while_the_entry_and_its_exports_keep_theirs():
    plain, friendly = renderings("llm_friendly_names")
    names = dict(friendly.names)
    present = identifiers(friendly.code)
    assert names
    assert {"llm_friendly_names", *friendly.arguments} <= present
    assert exported_names(friendly.code) == exported_names(plain.code)
    assert not any(name.startswith("__cpf_persistent_") for name in present)
    assert not set(names.values()) & present  # every respelled name is gone
    assert all(re.fullmatch(r"(arr|sym|f)\d+(_buf|_cap|_free|_idx|_size)?", short) for short in names)
    # Reverting the names gives back identifiers of the rendering only.
    assert identifiers(Compacted(friendly.code, names).revert(friendly.code)) <= identifiers(plain.code)


def test_a_device_dialect_is_refused():
    with pytest.raises(ValueError, match="host unit"):
        render(staged_stencil.to_sdfg(simplify=True), language="hip", llm_friendly=True, check_compiles=False)


def test_preprocessor_lines_strings_and_pastes_are_never_respelled_but_pragmas_are():
    """An include after a comment, a condition on a predefined macro, a multi-line macro and a string keep their
    text; an OpenMP clause names the code's variable, so it follows the code."""
    text = (
        "// unit\n#include <stdint.h>\n#if defined(__NVCOMPILER)\n#define SQ(x) \\\n  ((x) * (x))\n#endif\n"
        'static double h_tmp[4];\nconst char *s = "h_tmp";\n#pragma omp parallel for reduction(+:h_tmp)\n'
    )
    got = respell(text, {"h_tmp": "arr0", "x": "sym0", "stdint": "arr1", "__NVCOMPILER": "t0"})
    assert got == text.replace("double h_tmp", "double arr0").replace("+:h_tmp", "+:arr0")


class FakeSdfg:
    """Just what :func:`compact` reads of an SDFG: a name, and containers to offer for renaming."""

    def __init__(self, name: str, arrays: list[str]) -> None:
        self.name = name
        self.arrays = dict.fromkeys(arrays)
        self.symbols: dict[str, None] = {}

    def all_sdfgs_recursive(self) -> list["FakeSdfg"]:
        return [self]

    def all_nodes_recursive(self) -> list[tuple[object, object]]:
        return []

    def all_control_flow_regions(self) -> list[object]:
        return []


def test_names_a_directive_mentions_or_the_unit_exports_are_kept():
    code = (
        "#define USE_tmp_vals 1\n#if USE_tmp_vals\n#endif\nstatic double tmp_vals[3];\ndouble shared_out[3];\n"
        "void stencil_kernel(double *x) { x[0] = tmp_vals[0] + shared_out[0]; }\n"
    )
    sdfg = FakeSdfg("stencil_kernel", ["tmp_vals", "shared_out", "stencil_kernel"])
    result = compact(code, sdfg, {"x"})  # pyright: ignore[reportArgumentType] -- the fake has what compact reads
    assert "tmp_vals" in result.names.values()  # a directive names USE_tmp_vals, not tmp_vals
    assert "shared_out" not in result.names.values() and "stencil_kernel" not in result.names.values()
    guarded = code.replace("#if USE_tmp_vals", "#if defined(tmp_vals)")
    result = compact(guarded, sdfg, {"x"})  # pyright: ignore[reportArgumentType]
    assert "tmp_vals" not in result.names.values()


def test_derived_short_names_are_new_to_the_unit():
    code = (
        "static inline long tile_a_size(long n) { return n; }\n"
        "void entry_kernel(double *tile_buf, long arr0_size) { tile_buf[tile_a_size(arr0_size) - 1] = 0; }\n"
    )
    result = compact(code, FakeSdfg("entry_kernel", ["tile_a"]), {"tile_buf", "arr0_size"})  # pyright: ignore
    assert result.names.get("arr1_size") == "tile_a_size"  # arr0_size is the parameter's, so arr0 is skipped
    assert "arr0_size" in identifiers(result.code)


@pytest.mark.parametrize(
    "item",
    [
        "static int helper_fn(int c) {\n#if 1\n  if (c) {\n#else\n  if (!c) {\n#endif\n    return 1; }\n  return 0;\n}\n",
        "__attribute__((destructor)) static void release_buffer(void) {}\n",
        "static __attribute__((constructor)) void init_buffer(void) {}\n",
        "typedef int staticky;\nstaticky exported_global;\n",
    ],
)
def test_only_plain_unreferenced_static_definitions_are_dropped(item: str):
    code = item + "void entry_kernel(void) {}\n"
    assert drop_unused(code) == code
    assert drop_unused("static int unused_helper(void) { return 1; }\n" + code).lstrip("\n") == code
