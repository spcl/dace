# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF never serializes an accumulation behind ``omp critical``, and argmax / argmin reduce through declared pairs.

A conflicting write-conflict resolution takes an OpenMP atomic in every form: ``atomic update`` for the arithmetic
and bitwise operators, ``atomic compare`` for min / max, ``atomic write`` for a logical fold, a compare-and-swap loop
for any other fold, and one ``atomic update`` per component for a complex sum. An ``ArgReduce`` names the
``arg_max_pair`` / ``arg_min_pair`` reductions CPF declares over its (value, index) pair, ties to the lower index.
"""

import numpy as np
import pytest

import dace
from dace.codegen import cpf
from dace.libraries.standard.nodes import ArgReduce
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N", dace.int64)
SIZE = 512


@dace.program
def scatter(idx: dace.int64[N], val: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[idx[i]] += val[i]


def conflicting(name: str, wcr: str) -> dace.SDFG:
    """The scatter with its conflicting accumulation resolved by ``wcr`` instead of ``+``."""
    sdfg = scatter.to_sdfg(simplify=True)
    sdfg.name = name
    edges = [e for e, _ in sdfg.all_edges_recursive() if isinstance(e.data, dace.Memlet) and e.data.wcr is not None]
    assert edges, "the scatter has no conflicting accumulation to resolve"
    for edge in edges:
        edge.data.wcr = wcr
    return sdfg


def rendered_run(sdfg: dace.SDFG, language: str, arguments: dict) -> str:
    rendering = cpf.render(sdfg, language=language)
    call_standalone(build_standalone(rendering.code, sdfg.name, language=language), rendering.sdfg, arguments)
    return rendering.code


@pytest.mark.parametrize("language", ["c", "c++"])
@pytest.mark.parametrize(
    ("kind", "wcr", "ufunc", "form"),
    [
        ("max", "lambda a, b: max(a, b)", np.maximum, "cpf_atomic_max"),
        ("min", "lambda a, b: min(a, b)", np.minimum, "cpf_atomic_min"),
        ("sub", "lambda a, b: a - b", np.subtract, "omp atomic compare capture"),
    ],
)
def test_a_conflicting_accumulation_is_an_atomic_never_a_critical_section(language, kind, wcr, ufunc, form):
    sdfg = conflicting(f"cpf_conflict_{kind}_{'c' if language == 'c' else 'cpp'}", wcr)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, 16, size=SIZE).astype(np.int64)
    val = rng.random(SIZE)
    out = rng.random(SIZE)
    expected = out.copy()
    ufunc.at(expected, idx, val)

    code = rendered_run(sdfg, language, {"idx": idx, "val": val, "out": out, "N": SIZE})

    assert form in code and "critical" not in code, code
    assert np.allclose(out, expected, rtol=1e-12)


@pytest.mark.parametrize("language", ["c", "c++"])
def test_a_conflicting_complex_sum_is_one_atomic_per_component(language):
    sdfg = dace.SDFG(f"cpf_conflict_complex_{'c' if language == 'c' else 'cpp'}")
    sdfg.add_array("idx", [N], dace.int64)
    sdfg.add_array("val", [N], dace.complex128)
    sdfg.add_array("out", [N], dace.complex128)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "scatter",
        {"i": "0:N"},
        {"k": dace.Memlet("idx[i]"), "v": dace.Memlet("val[i]")},
        "o[k] = v",
        {"o": dace.Memlet("out[0:N]", wcr="lambda a, b: a + b", dynamic=True)},
        external_edges=True,
    )
    rng = np.random.default_rng(2)
    idx = rng.integers(0, 16, size=SIZE).astype(np.int64)
    val = rng.random(SIZE) + 1j * rng.random(SIZE)
    out = np.zeros(SIZE, dtype=np.complex128)
    expected = out.copy()
    np.add.at(expected, idx, val)

    code = rendered_run(sdfg, language, {"idx": idx, "val": val, "out": out, "N": SIZE})

    assert code.count("#pragma omp atomic update") >= 2 and "critical" not in code, code
    assert np.allclose(out, expected, rtol=1e-12)


def arg_reduce_sdfg(name: str, op: str, dtype: dace.typeclass) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N], dtype)
    sdfg.add_array("val", [1], dtype)
    sdfg.add_array("idx", [1], dace.int64)
    state = sdfg.add_state()
    node = ArgReduce("argreduce", op=op)
    state.add_node(node)
    state.add_edge(state.add_read("a"), None, node, "_in", dace.Memlet("a[0:N]"))
    state.add_edge(node, "_out_val", state.add_write("val"), None, dace.Memlet("val[0]"))
    state.add_edge(node, "_out_idx", state.add_write("idx"), None, dace.Memlet("idx[0]"))
    return sdfg


@pytest.mark.parametrize("language", ["c", "c++"])
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("dtype", [dace.float64, dace.int32, dace.complex128], ids=str)
def test_argmax_and_argmin_reduce_through_the_declared_pair_and_keep_the_first_extreme(language, op, dtype):
    sdfg = arg_reduce_sdfg(f"cpf_arg_{op}_{dtype.to_string()}_{'c' if language == 'c' else 'cpp'}", op, dtype)
    rng = np.random.default_rng(3)
    a = (rng.integers(0, 50, size=SIZE) + (1j * rng.integers(0, 50, size=SIZE) if dtype.is_complex() else 0)).astype(
        dtype.type
    )
    # Two copies of the extreme, far apart: the lower index must win however the threads split the range.
    extreme = a.max() if op == "max" else a.min()
    a[SIZE - 3] = extreme
    val, idx = np.zeros(1, dtype=dtype.type), np.zeros(1, dtype=np.int64)

    code = rendered_run(sdfg, language, {"a": a, "val": val, "idx": idx, "N": SIZE})

    assert f"reduction(arg_{op}_pair : __ar_best)" in code and "critical" not in code, code
    expected = int(np.argmax(a) if op == "max" else np.argmin(a))
    assert idx[0] == expected and val[0] == a[expected]
