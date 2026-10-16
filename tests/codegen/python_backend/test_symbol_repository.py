# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python execution preserves scoped proofs and their runtime preconditions."""

import numpy as np
import pytest
import sympy

import dace
from dace import dtypes, symbolic
from dace.codegen.py.python_target import PythonCodeGen
from dace.symbolic import Relation, RelationKind, Scale, Truth

N = dace.symbol("N")
K = dace.symbol("K")


@dace.program
def copy_prefix(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:K]:
        B[i] = A[i] + 1.0


@dace.program
def return_prefix(A: dace.float64[N]):
    return A[:K] + 1.0


def _checked_program(returns: bool = False) -> dace.SDFG:
    sdfg = (return_prefix if returns else copy_prefix).to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.symbol_repo.set_scale("K", Scale.SMALL)
    assert symbolic.ask(Relation(RelationKind.LT, K, N), sdfg.facts()) is Truth.TRUE
    return sdfg


@pytest.mark.parametrize("entry", ["call", "raw", "profiler", "initialize"])
def test_assumptions_checked_before_writes_and_initialization(entry: str):
    sdfg = _checked_program()
    sdfg.set_init_code("B[0] = -99", language=dtypes.Language.Python)
    compiled = sdfg.compile()
    a = np.arange(7, dtype=np.float64)
    b = np.full(7, 13.0)
    kwargs = dict(A=a, B=b, N=7, K=7)
    with pytest.raises(ValueError, match="K < N"):
        if entry == "call":
            compiled(**kwargs)
        elif entry == "raw":
            compiled._func(**kwargs)
        elif entry == "profiler":
            compiled._cfunc(None, **kwargs)
        else:
            compiled.initialize(**kwargs)
    np.testing.assert_array_equal(b, np.full(7, 13.0))
    assert not compiled._initialized


def test_assumptions_rechecked_on_every_call():
    compiled = _checked_program().compile()
    a = np.arange(11, dtype=np.float64)
    b = np.zeros(11)
    compiled(A=a, B=b, N=np.int64(11), K=np.int64(3))
    np.testing.assert_array_equal(b[:3], a[:3] + 1)
    before = b.copy()
    with pytest.raises(ValueError, match="K < N"):
        compiled(A=a, B=b, N=11, K=12)
    np.testing.assert_array_equal(b, before)
    compiled(A=a, B=b, N=11, K=5)
    np.testing.assert_array_equal(b[:5], a[:5] + 1)


def test_invalid_assumption_precedes_return_allocation(monkeypatch):
    compiled = _checked_program(returns=True).compile()

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("invalid call allocated a return buffer")

    monkeypatch.setattr(compiled, "_allocate_return_array", forbidden_allocation)
    with pytest.raises(ValueError, match="K < N"):
        compiled(A=np.arange(5, dtype=np.float64), K=6, N=5)


def test_valid_symbolic_return():
    compiled = _checked_program(returns=True).compile()
    a = np.arange(13, dtype=np.float64)
    np.testing.assert_array_equal(compiled(A=a, K=7, N=13), a[:7] + 1)


@pytest.mark.parametrize("wrap", [np.int64, np.asarray])
def test_numpy_integer_assumption_arithmetic_does_not_wrap(wrap):
    sdfg = dace.SDFG("assumption_arithmetic")
    sdfg.backend = dtypes.BackendLanguage.Python
    for name in ("K", "L"):
        sdfg.add_symbol(name, dace.int64, scale=Scale.SMALL)
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("B", [1], dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("write", {}, {"out"}, "out = K + L + N")
    state.add_edge(tasklet, "out", state.add_write("B"), None, dace.Memlet("B[0]"))
    relation = Relation(RelationKind.LT, K + dace.symbol("L"), N)
    assert symbolic.ask(relation, sdfg.facts()) is Truth.TRUE
    compiled = sdfg.compile()
    b = np.array([123], dtype=np.int64)
    with pytest.raises(ValueError, match="K \\+ L < N"):
        compiled._func(B=b, K=wrap(2**62), L=wrap(2**62), N=wrap(2**63 - 1))
    assert b[0] == 123


def test_nested_assumptions_use_outer_symbol_mapping():
    inner = _checked_program()
    outer = dace.SDFG("nested_assumption_mapping")
    outer.backend = dtypes.BackendLanguage.Python
    outer.add_symbol("M", dace.int64)
    outer.add_symbol("P", dace.int64)
    outer.add_array("A", ["M"], dace.float64)
    outer.add_array("B", ["M"], dace.float64)
    state = outer.add_state()
    node = state.add_nested_sdfg(inner, {"A"}, {"B"}, symbol_mapping={"N": "M", "K": "P + 1"})
    state.add_edge(state.add_read("A"), None, node, "A", dace.Memlet("A[0:M]"))
    state.add_edge(node, "B", state.add_write("B"), None, dace.Memlet("B[0:M]"))
    compiled = outer.compile()
    a, b = np.arange(9, dtype=np.float64), np.zeros(9)
    compiled(A=a, B=b, M=9, P=4)
    np.testing.assert_array_equal(b[:5], a[:5] + 1)
    before = b.copy()
    with pytest.raises(ValueError, match="P \\+ 1 < M"):
        compiled(A=a, B=b, M=9, P=8)
    np.testing.assert_array_equal(b, before)


def test_layout_proofs_use_only_supplied_facts():
    first, second = dace.SDFG("layout_positive"), dace.SDFG("layout_unknown")
    first.add_symbol("N", dace.int64, predicates=frozenset({symbolic.Predicate.POSITIVE}))
    second.add_symbol("N", dace.int64)
    expression = sympy.Abs(N)
    assert PythonCodeGen._provably_equal(expression, N, first.facts())
    assert not PythonCodeGen._provably_equal(expression, N, second.facts())
    assert PythonCodeGen._provably_equal(expression, N, first.facts())


def test_layout_simplification_records_each_graphs_runtime_assumption():
    for index in range(2):
        sdfg = dace.SDFG(f"layout_cached_{index}")
        sdfg.add_symbol("K", dace.int64, scale=Scale.SMALL)
        sdfg.add_symbol("N", dace.int64)
        assert PythonCodeGen._provably_equal(sympy.Max(K, N), N, sdfg.facts())
        assert Relation(RelationKind.LT, K, N) in sdfg.symbol_repo.runtime_assumptions


def test_fixed_tile_shape_uses_declared_constants_only():
    from dace.codegen.py.cutile_target import CuTilePythonCodeGen

    sdfg = dace.SDFG("tile_constants")
    sdfg.add_constant("W", 8)
    sdfg.add_array("tile", ["W", 4], dace.float64, transient=True)
    state = sdfg.add_state()
    entry, _ = state.add_map("tile_map", {"i": "0:32:8", "j": "0:16:4"})
    assert CuTilePythonCodeGen._resolve_single_tile_shape(entry, "tile", sdfg) == (8, 4)
    sdfg.add_array("dynamic_tile", ["N"], dace.float64, transient=True)
    with pytest.raises(ValueError, match="compile-time integer dimensions"):
        CuTilePythonCodeGen._resolve_single_tile_shape(entry, "dynamic_tile", sdfg)


@pytest.mark.parametrize(
    "kind,valid,invalid",
    [
        (RelationKind.LE, 3, 5),
        (RelationKind.EQ, 4, 3),
        (RelationKind.NE, 3, 4),
        (RelationKind.DIVIDES, 2, 3),
    ],
)
def test_declared_relations_checked_before_execution(kind: RelationKind, valid: int, invalid: int):
    sdfg = _checked_program()
    sdfg.symbol_repo.add_relation(Relation(kind, K, symbolic.pystr_to_symbolic(4)))
    compiled = sdfg.compile()
    a, b = np.arange(9, dtype=np.float64), np.zeros(9)
    compiled(A=a, B=b, N=9, K=valid)
    before = b.copy()
    with pytest.raises(ValueError, match="requires"):
        compiled(A=a, B=b, N=9, K=invalid)
    np.testing.assert_array_equal(b, before)


def test_nonnegative_repository_constraint_checked_before_initialization():
    sdfg = _checked_program()
    sdfg.set_symbol_assumptions("K", frozenset({symbolic.Predicate.NONNEGATIVE}))
    sdfg.set_init_code("B[0] = -99", language=dtypes.Language.Python)
    compiled = sdfg.compile()
    b = np.zeros(9)
    with pytest.raises(ValueError, match="0 <= K"):
        compiled(A=np.arange(9, dtype=np.float64), B=b, N=9, K=-1)
    np.testing.assert_array_equal(b, np.zeros(9))
    assert not compiled._initialized


def test_zero_divisor_matches_repository_divisibility():
    sdfg = dace.SDFG("zero_divisor_assumption")
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol("K", dace.int64)
    sdfg.add_symbol("N", dace.int64)
    sdfg.symbol_repo.add_relation(Relation(RelationKind.DIVIDES, K, N))
    sdfg.add_array("B", [1], dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("copy", {}, {"out"}, "out = K + N")
    state.add_edge(tasklet, "out", state.add_write("B"), None, dace.Memlet("B[0]"))
    compiled = sdfg.compile()
    output = np.full(1, 123, dtype=np.int64)
    compiled(B=output, K=0, N=0)
    np.testing.assert_array_equal(output, [0])
    with pytest.raises(ValueError, match="K divides N"):
        compiled(B=output, K=0, N=1)
    np.testing.assert_array_equal(output, [0])
