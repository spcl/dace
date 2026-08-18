# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Structural regressions for the autodiff symbolic/nested-SDFG audit.

Each test pins an invariant that used to hold only by accident: how a subset dimension is parsed,
what kind of symbol reaches a differentiated expression, and what a backward nested SDFG node
declares about its symbols and connectors.
"""
import typing
import warnings

import numpy as np
import pytest
import sympy

import dace
import dace.sdfg.nodes as nd
from dace import symbolic
from dace.sdfg.state import LoopRegion

import dace.autodiff.utils as ad_utils
import dace.autodiff.data_forwarding.store as ad_store
from dace.autodiff import add_backward_pass
from dace.autodiff.backward_pass_generator import BackwardPassGenerator

N = dace.symbol("N", dtype=dace.int64)


class BoundsGeneratorStub:
    """The two attributes ``get_symbol_upper_bound_from_loop`` reads off the generator."""

    def __init__(self, sdfg: dace.SDFG):
        self.sdfg = sdfg
        self.interstate_symbols: dict[str, str] = {}


def counting_loop() -> tuple[dace.SDFG, LoopRegion]:
    sdfg = dace.SDFG("stored_bounds")
    loop = LoopRegion("loop", condition_expr="i < 10", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(loop)
    loop.add_state("body", is_start_block=True)
    return sdfg, loop


@dace.program
def scale_and_sum(X: dace.float64[N], Y: dace.float64[N], S: dace.float64[1]):
    Y[:] = X * 2.0
    S[0] = np.sum(Y)


def differentiated_sdfg() -> tuple[dace.SDFG, list[warnings.WarningMessage]]:
    """Backward pass over a symbolically shaped program, with the warnings it raised.

    Only the backward generation is recorded: the Python frontend raises the same set-connector
    warning while parsing, and that is not this module's business.
    """
    sdfg = scale_and_sum.to_sdfg(simplify=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        add_backward_pass(sdfg=sdfg, outputs=["S"], inputs=["X"], simplify=False)
    return sdfg, caught


def nested_sdfg_nodes(sdfg: dace.SDFG) -> list[tuple[nd.NestedSDFG, dace.SDFGState]]:
    return [(node, parent) for node, parent in sdfg.all_nodes_recursive() if isinstance(node, nd.NestedSDFG)]


# Issue 1: subset dimensions are parsed with DaCe's parser, not sympy's.
def test_stored_dimension_parse_keeps_integer_division():
    sdfg, loop = counting_loop()
    generator = BoundsGeneratorStub(sdfg)

    # The two parsers disagree on `//`: DaCe builds an int_floor node, sympify a rational division
    # that folds `i//i` to 1 and loses the loop index entirely.
    assert isinstance(symbolic.pystr_to_symbolic("i//2"), symbolic.int_floor)
    assert sympy.sympify("i//2").has(sympy.floor)
    assert sympy.sympify("i//i").free_symbols == set()

    for dimension in ("i", "i//2", "i//i", "2*i"):
        bound, loop_index = ad_store.get_symbol_upper_bound_from_loop(generator, dimension, [loop])
        assert loop_index == "i"
        assert bound == 10


# Issue 2: no bare sympy symbol reaches a caller of the symbolic-differentiation path.
def test_code_to_exprs_yields_dace_symbols_with_the_sdfg_dtype():
    sdfg = dace.SDFG("tasklet_symbols")
    sdfg.add_symbol("N", dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("t", {"a": dace.float64, "b": dace.float64}, {"c": dace.float64}, "c = a * b + N")

    exprs, indexed = ad_utils.code_to_exprs(tasklet.code.as_string, tasklet, sdfg.symbols)

    assert indexed == {}
    free = {str(s): s for s in exprs["c"].free_symbols}
    assert set(free) == {"a", "b", "N"}
    for sym in free.values():
        assert isinstance(sym, symbolic.symbol), f"{sym} is a bare sympy symbol"
    assert free["N"].dtype == dace.int64
    assert free["a"].dtype == dace.float64


def test_differentiation_target_is_the_instance_inside_the_expression():
    x_in_expr = symbolic.symbol("x", dace.float64)
    expr = x_in_expr * 3

    resolved = ad_utils.resolve_differentiation_target(expr, "x", None)
    assert resolved is x_in_expr
    assert expr.diff(resolved) == 3

    # A re-minted symbol is what the fix removes: it compares unequal and differentiates to zero.
    assert expr.diff(sympy.Symbol("x")) == 0

    # A connector that does not occur still yields a DaCe symbol, and a zero derivative.
    missing = ad_utils.resolve_differentiation_target(expr, "y", None)
    assert isinstance(missing, symbolic.symbol)
    assert expr.diff(missing) == 0


def test_indexed_differentiation_target_is_the_instance_inside_the_expression():
    base = sympy.IndexedBase("A")
    expr = base[ad_utils.index_symbol("i", dace.int64)] * 2

    resolved = ad_utils.resolve_differentiation_target(expr, "A", ["i"])
    assert resolved is next(iter(expr.atoms(sympy.Indexed)))
    assert expr.diff(resolved) == 2

    # Rebuilding the access from bare names produces a KroneckerDelta rather than the derivative.
    assert expr.diff(base[sympy.Idx("i")]).has(sympy.KroneckerDelta)


# Issue 3: every backward nested SDFG declares its symbols explicitly.
@pytest.mark.autodiff
def test_backward_nested_sdfgs_declare_their_symbol_mapping():
    sdfg, _ = differentiated_sdfg()
    nested = nested_sdfg_nodes(sdfg)
    assert nested, "expected the backward pass to build at least one nested SDFG"

    for node, parent in nested:
        required = node.sdfg.free_symbols - {"NoneSymbol"}
        assert required, f"nested SDFG {node.label} carries no symbol; the test program lost its symbolic shape"
        assert required <= set(node.symbol_mapping), (f"nested SDFG {node.label} leaves "
                                                      f"{sorted(required - set(node.symbol_mapping))} unmapped")
        for name in required:
            declared = parent.sdfg.symbols.get(name)
            if declared is None:
                continue
            mapped = node.symbol_mapping[name]
            assert str(mapped) == name
            assert isinstance(mapped, symbolic.symbol), f"{node.label}:{name} mapped to a bare {type(mapped)}"
            assert mapped.dtype == declared, f"{node.label}:{name} mapped as {mapped.dtype}, parent says {declared}"


def test_backward_symbol_mapping_carries_the_parent_dtype():
    parent = dace.SDFG("parent")
    parent.add_symbol("N", dace.int64)
    parent_state = parent.add_state()

    child = dace.SDFG("child")
    child.add_array("data", [dace.symbol("N", dace.int32)], dace.float64)
    child.add_state().add_access("data")

    mapping = ad_utils.backward_symbol_mapping(child, parent_state)
    assert set(mapping) == {"N"}
    assert isinstance(mapping["N"], symbolic.symbol)
    assert mapping["N"].dtype == dace.int64


@pytest.mark.autodiff
def test_backward_pass_does_not_build_nested_sdfgs_from_sets():
    sdfg, caught = differentiated_sdfg()

    offenders = [str(w.message) for w in caught if "sets as inputs" in str(w.message)]
    assert not offenders, offenders

    nested = nested_sdfg_nodes(sdfg)
    assert nested
    for node, _ in nested:
        assert list(node.in_connectors) == sorted(node.in_connectors)
        assert list(node.out_connectors) == sorted(node.out_connectors)


# Issues 5 and 6: the forward references in annotations name a class that exists.
def test_backward_pass_generator_forward_references_resolve():
    annotated = [
        ad_store.get_symbol_upper_bound_from_loop,
        ad_store.resolve_overwrite_with_store,
        ad_store.store_data,
        ad_store.connect_stored_data_to_target,
    ]
    for fn in annotated:
        # localns stands in for the ``if TYPE_CHECKING:`` import, which is absent at runtime by design.
        hints = typing.get_type_hints(fn, localns={"BackwardPassGenerator": BackwardPassGenerator})
        assert hints["bwd_generator"] is BackwardPassGenerator
