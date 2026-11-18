# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The LLM-friendly spelling of a CPF unit computes what the rendering computes, under short names it can revert."""

import re

import numpy as np

import dace
from dace.codegen.cpf import render
from dace.codegen.cpf_compact import compact, tokens
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


def run(rendering, code: str, name: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    a, b = rng.random(n), np.zeros(n)
    call_standalone(build_standalone(code, name, language="c"), rendering.sdfg, {"a": a, "b": b, "N": n, "steps": 3})
    return a, b


def test_the_compacted_unit_computes_the_same_values_in_fewer_characters():
    sdfg = staged_stencil.to_sdfg(simplify=True)
    sdfg.name = "llm_friendly_stencil"
    rendering = render(sdfg, language="c")
    compacted = compact(rendering)
    assert len(compacted.code) < len(rendering.code)
    for n in (1, 7, 64):
        want = run(rendering, rendering.code, "llm_friendly_stencil_ref", n)
        got = run(rendering, compacted.code, "llm_friendly_stencil_compact", n)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])


def test_generated_names_shorten_parameters_stay_and_the_map_reverts_them():
    sdfg = staged_stencil.to_sdfg(simplify=True)
    sdfg.name = "llm_friendly_names"
    rendering = render(sdfg, language="c")
    compacted = compact(rendering)
    names = identifiers(compacted.code)
    assert set(rendering.arguments) <= names  # the entry point's parameters keep their names
    assert not any(name.startswith("__cpf_persistent_") for name in names)
    assert not {name for name in compacted.names.values() if name in names}  # every respelled name is gone
    assert all(re.fullmatch(r"(arr|sym|t|f)\d+(_buf|_cap|_free|_idx|_size)?", short) for short in compacted.names)
    # Reverting the names gives back every identifier of the rendering that the unit still uses.
    assert identifiers(compacted.revert(compacted.code)) <= identifiers(rendering.code)
