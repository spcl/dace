# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``ipow`` symbolic Function and the ``RelaxIntegerPowers`` pass."""

import numpy as np
import pytest
import sympy

import dace
from dace.symbolic import Predicate
from dace import data, symbolic, symbolic_engine
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SymbolResolver
from dace.symbolic import ipow, pystr_to_symbolic, symstr
from dace.transformation.passes.relax_integer_powers import RelaxIntegerPowers, relaxed_exponent


def ipow_count(sdfg: dace.SDFG) -> int:
    """Count ``ipow`` occurrences across descriptors, map ranges and memlet subsets."""

    def atoms(expr):
        return len(expr.atoms(ipow)) if isinstance(expr, symbolic.SymbolicBasic) else 0

    total = 0
    for sd in sdfg.all_sdfgs_recursive():
        for desc in sd.arrays.values():
            if isinstance(desc, data.Array):
                total += sum(atoms(e) for e in (*desc.shape, *desc.strides, desc.total_size))
        for state in sd.states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    total += sum(atoms(x) for rng in node.map.range.ranges for x in rng)
            for edge in state.edges():
                if edge.data is not None and isinstance(edge.data.subset, dace.subsets.Range):
                    total += sum(atoms(x) for rng in edge.data.subset.ranges for x in rng)
    return total


def test_ipow_lowers_to_cpp_ipow():
    R, K = (symbolic.symbol(s, positive=True, integer=True) for s in ("R", "K"))
    assert "dace::math::ipow(R, K)" in symstr(ipow(R, K), cpp_mode=True)


def test_ipow_roundtrips_through_serialization():
    R, K, i = (symbolic.symbol(s) for s in ("R", "K", "i"))
    e = R * ipow(R, K - i - 1) + ipow(R, K - i - 1)
    back = pystr_to_symbolic(str(e))
    assert symstr(e, cpp_mode=True) == symstr(back, cpp_mode=True)
    assert any(symbolic.head_name(a) == "ipow" for a in symbolic_engine.preorder_traversal(back))


def test_ipow_survives_property_json_roundtrip_and_folds():
    P = symbolic.symbol("P")
    e = 64 * ipow(P, 2)
    back = symbolic.deserialize_symbolic(symbolic.serialize_symbolic(e))
    assert any(symbolic.head_name(a) == "ipow" for a in symbolic_engine.preorder_traversal(back))
    assert int(symbolic.evaluate(back, {P: 4})) == 64 * 16


def test_ipow_folds_constant_power():
    assert ipow(sympy.Integer(2), sympy.Integer(10)) == 1024


def test_interval_proves_radix_decomposition():
    K = symbolic.symbol("K")
    i = symbolic.symbol("i")
    positive = symbolic.predicate_relation(symbolic.Predicate.POSITIVE, K)
    in_range = (
        symbolic.Relation(symbolic.RelationKind.LE, sympy.Integer(0), i),
        symbolic.Relation(symbolic.RelationKind.LE, i, K - 1),
    )

    def prove(exp, relations):
        return symbolic.provably_nonnegative(exp, symbolic.Facts(relations, frozenset({"K", "i"})))

    assert prove(K - i - 1, (positive, *in_range)) is True  # bottoms out at 0 when i = K-1
    assert prove(i, (positive, *in_range)) is True
    assert prove(K, (positive,)) is True  # positive size symbol


def test_interval_refuses_unbounded_iterator():
    K = symbolic.symbol("K")
    i = symbolic.symbol("i")
    facts = symbolic.Facts((symbolic.predicate_relation(symbolic.Predicate.POSITIVE, K),), frozenset({"K", "i"}))
    # Without a range for i, K - i - 1 could be negative -> must not relax.
    assert symbolic.provably_nonnegative(K - i - 1, facts) is False


def test_relaxes_pow_inside_loop():
    """A radix subscript ``R**(K-i-1)`` in a memlet under ``for i in range(K)``
    relaxes via the iterator range the loop binds during the descent."""
    R = dace.symbol("R", positive=True, integer=True)
    K = dace.symbol("K", positive=True, integer=True)

    sdfg = dace.SDFG("loop")
    sdfg.add_array("x", [R**K], dace.float64)
    loop = LoopRegion(
        label="loop", condition_expr="i < K", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1"
    )
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("set", {}, {"o"}, "o = 1.0")
    body.add_edge(tasklet, "o", body.add_write("x"), None, dace.Memlet("x[R**(K - i - 1) - 1]"))

    assert RelaxIntegerPowers().apply_pass(sdfg, {}) is not None
    assert ipow_count(sdfg) > 0  # R**(K-i-1) proven >= 0 from i in [0, K-1]


def test_relaxes_pow_inside_map_and_nested_sdfg():
    """An indirect access ``a[b[i]]`` inside a map lowers to a nested SDFG; a radix
    power there must relax with the map iterator carried through the symbol map."""
    R = dace.symbol("R", positive=True, integer=True)
    M = dace.symbol("M", positive=True, integer=True)

    @dace.program
    def prog(a: dace.float64[R**M], b: dace.int64[M]):
        for i in dace.map[0:M]:
            a[b[i] + R ** (M - i - 1) - 1] = 2.0

    sdfg = prog.to_sdfg(simplify=False)
    assert any(isinstance(n, nodes.NestedSDFG) for n, _ in sdfg.all_nodes_recursive())
    assert RelaxIntegerPowers().apply_pass(sdfg, {}) is not None
    assert ipow_count(sdfg) > 0


def test_relaxes_under_dynamic_map_symbol():
    """A map whose range is a dynamic input (a data-dependent symbol) is handled; a
    power over a provably-positive symbol still relaxes."""
    R = dace.symbol("R", positive=True, integer=True)
    N = dace.symbol("N", positive=True, integer=True)

    sdfg = dace.SDFG("dyn")
    sdfg.add_symbol(R)
    sdfg.add_symbol(N)
    sdfg.add_array("x", [R**N], dace.float64)
    sdfg.add_array("bound", [1], dace.int64)
    state = sdfg.add_state()
    me, mx = state.add_map("m", dict(j="0:s"))  # dynamic upper bound symbol s
    me.add_in_connector("s")
    state.add_edge(state.add_read("bound"), None, me, "s", dace.Memlet("bound[0]"))
    t = state.add_tasklet("w", {}, {"o"}, "o = 1.0")
    state.add_edge(me, None, t, None, dace.Memlet())
    w = state.add_write("x")
    state.add_memlet_path(t, mx, w, src_conn="o", memlet=dace.Memlet("x[R**N - 1]"))

    # Must not crash on the dynamic-range symbol, and must relax the R**N size/index.
    assert RelaxIntegerPowers().apply_pass(sdfg, {}) is not None
    assert ipow_count(sdfg) > 0


def test_refuses_unprovable_and_negative_exponents():
    """A sign-unknown exponent and a provably-negative one stay on the ``pow`` path."""
    R = dace.symbol("R", positive=True, integer=True)
    K = dace.symbol("K", positive=True, integer=True)
    M = dace.symbol("M", integer=True)  # no sign assumption

    sdfg = dace.SDFG("unprovable")
    sdfg.add_array("u", [R**M], dace.float64)
    sdfg.add_state().add_access("u")
    RelaxIntegerPowers().apply_pass(sdfg, {})
    assert ipow_count(sdfg) == 0
    assert sdfg.arrays["u"].shape[0].has(sympy.Pow)  # R**M unchanged

    assert relaxed_exponent(-K, symbolic.Facts.none()) is None
    assert relaxed_exponent(sympy.Rational(1, 2), symbolic.Facts.none()) is None


def test_end_to_end_complex_power_shape_compiles():
    """A ``complex128`` array whose shape ``R**K`` is a symbolic power (the
    stockham-fft shape) must relax to ``ipow`` -- ``pow`` would be an illegal
    ``double`` array size / map bound -- and stay bit-exact with NumPy."""
    R = dace.symbol("R", positive=True, integer=True)
    K = dace.symbol("K", positive=True, integer=True)

    sdfg = dace.SDFG("power_shape")
    sdfg.add_symbol(R)
    sdfg.add_symbol(K)
    sdfg.add_array("x", [R**K], dace.complex128)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "scale",
        dict(j=f"0:{R**K}"),
        {"a": dace.Memlet("x[j]")},
        "b = a * 2.0",
        {"b": dace.Memlet("x[j]")},
        external_edges=True,
    )

    assert RelaxIntegerPowers().apply_pass(sdfg, {}) is not None
    assert ipow_count(sdfg) > 0

    Rv, Kv = 2, 4
    x = (np.arange(1, Rv**Kv + 1) + 1j * np.arange(1, Rv**Kv + 1)).astype(np.complex128)
    ref = x * 2.0
    sdfg(x=x, R=Rv, K=Kv)
    assert np.allclose(x, ref)


@pytest.mark.parametrize("bad", [sympy.Integer(-1), sympy.Integer(-7), sympy.Rational(-1, 2), sympy.Float(-0.5)])
def test_ipow_rejects_a_negative_constant_exponent_integral_or_not(bad):
    """A negative constant is a reciprocal, so it is a ``Pow``; the sign test does not ask for an integer."""
    with pytest.raises(ValueError):
        ipow(symbolic.symbol("R", positive=True, integer=True), bad)


def test_ipow_leaves_an_exponent_of_unknown_sign_to_the_caller():
    base = symbolic.symbol("R", positive=True, integer=True)
    unknown = symbolic.symbol("M", integer=True)
    assert isinstance(ipow(base, unknown), ipow)
    with pytest.raises(ValueError):
        ipow(base, unknown).subs(unknown, -1)


def test_a_symbolic_exponent_of_unknown_sign_is_not_relaxed_by_the_pass():
    R = dace.symbol("R", positive=True, integer=True)
    M = dace.symbol("M", integer=True)
    sdfg = dace.SDFG("unknown_sign_exponent")
    sdfg.add_array("u", [R**M], dace.float64)
    sdfg.add_state().add_access("u")
    RelaxIntegerPowers().apply_pass(sdfg, {})
    assert ipow_count(sdfg) == 0


def test_an_unsigned_exponent_is_relaxed_by_the_pass():
    R = dace.symbol("R", positive=True, integer=True)
    M = dace.symbol("M", dace.uint32)
    sdfg = dace.SDFG("unsigned_exponent")
    sdfg.add_symbol("M", dace.uint32)
    sdfg.add_array("u", [R**M], dace.float64)
    sdfg.add_state().add_access("u")
    RelaxIntegerPowers().apply_pass(sdfg, {})
    assert ipow_count(sdfg) > 0


def test_loop_range_direction_from_stride_sign():
    """The iterator range of a loop is ordered by the stride sign, and refused when the sign is
    unknown -- a wrong direction guess would relax a negative exponent."""

    def proves(update, init, cond, low, high):
        sdfg = dace.SDFG("stride_sign")
        sdfg.add_symbol("K", dace.int64)
        sdfg.add_symbol("sstep", dace.int64)
        loop = LoopRegion("L", condition_expr=cond, loop_var="i", initialize_expr=init, update_expr=update)
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state("body", is_start_block=True)
        facts = SymbolResolver().facts_at(loop)
        i = symbolic.symbol("i")
        return symbolic.provably_le(pystr_to_symbolic(low), i, facts) and symbolic.provably_le(
            i, pystr_to_symbolic(high), facts
        )

    assert proves("i = i + 1", "i = 0", "i < K", "0", "K - 1")  # ascending
    assert proves("i = i - 1", "i = K", "i > 0", "1", "K")  # descending
    assert not proves("i = i + sstep", "i = K", "i > 0", "1", "K")  # unknown-sign stride -> no trusted range


def test_refuses_pow_under_unknown_sign_stride():
    """The stride sign is the only thing that changes between these two loops: ``K - p`` is
    provably >= 0 over an ascending sweep, so it relaxes; under an unknown-sign stride there is
    no trusted range, so it must stay ``pow`` (if the stride were negative it would go negative)."""
    R = dace.symbol("R", positive=True, integer=True)
    K = dace.symbol("K", positive=True, integer=True)
    p = dace.symbol("p", integer=True)
    sstep = dace.symbol("sstep", integer=True)  # unknown sign

    def count(name, update):
        sdfg = dace.SDFG(name)
        sdfg.add_array("x", [100], dace.float64)  # constant shape -> only the subscript carries a power
        sdfg.add_symbol(R)
        sdfg.add_symbol(K)
        loop = LoopRegion("L", condition_expr="p < K", loop_var="p", initialize_expr="p = 0", update_expr=update)
        sdfg.add_node(loop, is_start_block=True)
        body = loop.add_state("body", is_start_block=True)
        t = body.add_tasklet("set", {}, {"o"}, "o = 1.0")
        body.add_edge(t, "o", body.add_write("x"), None, dace.Memlet("x[R**(K - p)]"))
        RelaxIntegerPowers().apply_pass(sdfg, {})
        return ipow_count(sdfg)

    assert count("ascending", "p = p + 1") > 0  # p in [0, K-1] -> K-p in [1, K] -> relax
    assert count("unknown", "p = p + sstep") == 0  # unknown-sign stride -> stays pow


def test_descending_loop_still_relaxes():
    """A radix subscript proven non-negative over a descending loop's range relaxes (direction
    is handled, not just refused)."""
    R = dace.symbol("R", positive=True, integer=True)
    K = dace.symbol("K", positive=True, integer=True)

    sdfg = dace.SDFG("descending")
    sdfg.add_symbol(R)
    sdfg.add_symbol(K)
    sdfg.add_array("x", [100], dace.float64)
    loop = LoopRegion("L", condition_expr="i > 0", loop_var="i", initialize_expr="i = K", update_expr="i = i - 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    t = body.add_tasklet("set", {}, {"o"}, "o = 1.0")
    body.add_edge(t, "o", body.add_write("x"), None, dace.Memlet("x[R**(K - i)]"))  # i in [1, K] -> K-i in [0, K-1]

    RelaxIntegerPowers().apply_pass(sdfg, {})
    assert ipow_count(sdfg) > 0


def test_loop_condition_off_by_one_not_relaxed():
    """The loop condition is evaluated one past the body range (``i = end + step``), so an
    exponent that uses the iterator must not relax there; a constant-exponent bound still does."""
    R = dace.symbol("R", positive=True, integer=True)
    K = dace.symbol("K", positive=True, integer=True)

    def build(cond):
        sdfg = dace.SDFG("cond")
        sdfg.add_symbol(R)
        sdfg.add_symbol(K)  # declared positive, so the pass knows K is a positive int
        sdfg.add_array("x", [R**K], dace.float64)
        loop = LoopRegion("L", condition_expr=cond, loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state("body", is_start_block=True)
        RelaxIntegerPowers().apply_pass(sdfg, {})
        return loop.loop_condition.as_string

    assert "ipow" not in build("i < R**(K - i - 1)")  # exponent uses i, negative at i = K
    assert "ipow" in build("i < R**K")  # constant exponent -> safe to relax


def test_relaxes_pow_inside_a_packed_product():
    """A stored shape product comes back from serialization unevaluated, in the order the evaluating
    constructor would give it. The relaxation must reach the powers inside it, or a ``double``
    ``pow`` stays in an int64 array size."""
    R = dace.symbol("R", positive=True, integer=True)
    K = dace.symbol("K", positive=True, integer=True)
    i = dace.symbol("i", positive=True, integer=True)

    sdfg = dace.SDFG("packed")
    sdfg.add_array("x", [R, R**i, R ** (K - i - 1)], dace.float64)  # the stockham stride-permutation shape
    loop = LoopRegion("L", condition_expr="i < K", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True).add_access("x")  # i in [0, K-1] -> K - i - 1 >= 0
    sdfg = dace.SDFG.from_json(sdfg.to_json())  # the route a packed product actually reaches the pass by

    packed = sdfg.arrays["x"].total_size
    assert packed.args == symbolic_engine.Mul(*packed.args).args  # loaded in canonical order

    assert RelaxIntegerPowers().apply_pass(sdfg, {}) is not None
    total = sdfg.arrays["x"].total_size
    assert len(total.atoms(ipow)) == 2  # both powers inside the product, not just the bare shape dims
    assert not total.atoms(sympy.Pow)  # no libm pow survives in an int64 size
    assert "dace::math::pow" not in symstr(total, cpp_mode=True)


def test_nested_sdfg_inherits_outer_sign_facts():
    """A nested SDFG an expansion mints spells its shapes bare (a reparse cannot know signs) --
    the sign facts are the outer SDFG's declared predicates. The facts have to travel by name across the nesting, else ``R**K`` inside declines for want of a sign and
    codegen emits a ``double`` ``pow`` loop bound."""
    bare_shape = [pystr_to_symbolic("R**K")]  # bare symbols, exactly as a reparsed store spells them

    inner = dace.SDFG("inner")
    inner.add_symbol("R", dace.int64)
    inner.add_symbol("K", dace.int64)
    inner.add_array("y", bare_shape, dace.float64)
    inner.add_state().add_access("y")
    # the whole point: no signs spelled here
    assert all(s.is_positive is None for s in inner.arrays["y"].free_symbols)

    R, K = dace.symbol("R"), dace.symbol("K")
    outer = dace.SDFG("outer")
    outer.add_symbol("R", dace.int64, predicates=frozenset({Predicate.POSITIVE}))
    outer.add_symbol("K", dace.int64, predicates=frozenset({Predicate.POSITIVE}))
    outer.add_array("y", [R**K], dace.float64)
    state = outer.add_state()
    nsdfg = state.add_nested_sdfg(inner, {}, {"y"}, symbol_mapping={"R": "R", "K": "K"})
    state.add_edge(nsdfg, "y", state.add_write("y"), None, dace.Memlet("y[0:R**K]"))

    assert RelaxIntegerPowers().apply_pass(outer, {}) is not None
    assert len(inner.arrays["y"].shape[0].atoms(ipow)) == 1
    assert not inner.arrays["y"].shape[0].atoms(sympy.Pow)


def test_a_nested_argument_with_no_access_node_is_relaxed_like_its_outer_descriptor():
    """A nested SDFG that only passes ``y`` on (its access is one level deeper) has no access node of ``y``;
    its descriptor must still be relaxed, or it no longer matches the parent's and validation refuses the
    connection (QE addusxx_g's ``ijtoh[ntyp, nhm, nhm]``)."""
    N = dace.symbol("N", dtype=dace.int64, integer=True, positive=True)

    def holder(name: str) -> dace.SDFG:
        sdfg = dace.SDFG(name)
        sdfg.add_symbol("N", dace.int64)
        sdfg.add_array("y", [N, N, N], dace.float64)
        return sdfg

    leaf = holder("leaf")
    leaf.add_state().add_access("y")
    middle = holder("middle")  # only a nested SDFG uses y here: no access node of y at this level
    mid_state = middle.add_state()
    inner_node = mid_state.add_nested_sdfg(leaf, {"y"}, {}, symbol_mapping={"N": "N"})
    mid_state.add_edge(mid_state.add_read("y"), None, inner_node, "y", dace.Memlet("y[0:N, 0:N, 0:N]"))
    mid_state.remove_node(next(n for n in mid_state.data_nodes() if n.data == "y"))
    outer = holder("outer")
    state = outer.add_state()
    nsdfg = state.add_nested_sdfg(middle, {"y"}, {}, symbol_mapping={"N": "N"})
    state.add_edge(state.add_read("y"), None, nsdfg, "y", dace.Memlet("y[0:N, 0:N, 0:N]"))
    assert not any(n.data == "y" for n in mid_state.data_nodes())

    RelaxIntegerPowers().apply_pass(outer, {})
    assert symstr(middle.arrays["y"].strides[0]) == symstr(outer.arrays["y"].strides[0])
    assert symstr(middle.arrays["y"].total_size) == symstr(outer.arrays["y"].total_size)


def test_int64_power_compiles_as_an_integer():
    """A power of int64 symbols the pass proves non-negative lowers to ``ipow`` and stays integral: as a
    double it is neither a legal OpenMP controlling predicate nor a pointer offset."""
    R = dace.symbol("R", dtype=dace.int64)
    K = dace.symbol("K", dtype=dace.int64, nonnegative=True)

    @dace.program
    def tester(A: dace.float64[R**K]):
        for i in dace.map[0 : R**K]:
            A[i] = 1.0

    a = np.zeros(8)
    tester(a, R=2, K=3)
    assert np.allclose(a, 1.0)


def test_interstate_assignment_keeps_array_access_and_evaluation_order():
    """An interstate assignment reading an array is rewritten only at its ``**``: the array access stays a
    subscript (not a SymPy ``Subscript(...)`` call) and the floating-point operations keep their written order."""
    sdfg = dace.SDFG("assign")
    sdfg.add_array("rho", [4], dace.float64)
    sdfg.add_symbol("lam", dace.float64)
    first = sdfg.add_state("first", is_start_block=True)
    second = sdfg.add_state("second")
    edge = sdfg.add_edge(
        first, second, dace.InterstateEdge(assignments={"v": "((4.0 * lam) / rho[1]) * (lam ** 2.0) + lam ** (- 0.75)"})
    )

    assert RelaxIntegerPowers().apply_pass(sdfg, {}) == 1
    assert edge.data.assignments["v"] == "((((4.0 * lam) / rho[1]) * ipow(lam, 2)) + (lam ** (- 0.75)))"


if __name__ == "__main__":
    test_ipow_lowers_to_cpp_ipow()
    test_ipow_roundtrips_through_serialization()
    test_ipow_survives_property_json_roundtrip_and_folds()
    test_ipow_folds_constant_power()
    test_interval_proves_radix_decomposition()
    test_interval_refuses_unbounded_iterator()
    test_relaxes_pow_inside_loop()
    test_relaxes_pow_inside_map_and_nested_sdfg()
    test_relaxes_under_dynamic_map_symbol()
    test_refuses_unprovable_and_negative_exponents()
    test_end_to_end_complex_power_shape_compiles()
    for bad in (sympy.Integer(-1), sympy.Integer(-7), sympy.Rational(-1, 2), sympy.Float(-0.5)):
        test_ipow_rejects_a_negative_constant_exponent_integral_or_not(bad)
    test_ipow_leaves_an_exponent_of_unknown_sign_to_the_caller()
    test_a_symbolic_exponent_of_unknown_sign_is_not_relaxed_by_the_pass()
    test_an_unsigned_exponent_is_relaxed_by_the_pass()
    test_loop_range_direction_from_stride_sign()
    test_refuses_pow_under_unknown_sign_stride()
    test_descending_loop_still_relaxes()
    test_loop_condition_off_by_one_not_relaxed()
    test_relaxes_pow_inside_a_packed_product()
    test_nested_sdfg_inherits_outer_sign_facts()
    test_a_nested_argument_with_no_access_node_is_relaxed_like_its_outer_descriptor()
    test_int64_power_compiles_as_an_integer()
    test_interstate_assignment_keeps_array_access_and_evaluation_order()
