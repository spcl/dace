# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``RelaxIntegerPowers`` -- lower ``base ** exp`` to ``ipow`` where the exponent is a non-negative integer.

DaCe's symbolic C++ printer emits ``dace::math::pow`` (libm, ``double``) for a non-constant exponent, which is
illegal where an integer is required -- an array size, a subscript, or a loop bound.

``Pow(base, exp) -> ipow(base, exp)`` whenever ``exp`` is a provable non-negative integer: a non-negative integer
constant, an integer-valued float literal, or a symbolic integer proven ``>= 0`` by interval analysis over the
enclosing iterator ranges (``K - i - 1`` with ``for i in range(K)``).
"""
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

import sympy

from dace import SDFG, data, dtypes, subsets, symbolic
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.symbolic import equalize_symbol, ipow
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.analysis import loop_analysis

#: Inclusive ``(low, high)`` of an iterator.
Bounds = tuple[symbolic.SymbolicType, symbolic.SymbolicType]
#: The live iteration ranges, by iterator name.
Ranges = dict[str, Bounds]


@dataclass(frozen=True, slots=True)
class SignFacts:
    """Names an SDFG declares positive, non-negative or integer, kept by name so that a same-named symbol that lost
    its assumptions is still covered."""
    positive: frozenset = frozenset()
    nonnegative: frozenset = frozenset()
    integer: frozenset = frozenset()

    @staticmethod
    def of_sdfg(sdfg: SDFG) -> 'SignFacts':
        # The assumptions live on the symbol objects in the descriptors; ``sdfg.free_symbols`` only has names.
        syms = set()
        for desc in sdfg.arrays.values():
            if isinstance(desc, data.Array):
                syms |= desc.free_symbols
        # An unsigned dtype is a sign fact sympy's assumptions do not carry
        unsigned = frozenset(name for name, dtype in sdfg.symbols.items()
                             if type(dtype) is dtypes.typeclass and dtype.as_numpy_dtype().kind == 'u')
        nonnegative = frozenset(s.name for s in syms if s.is_nonnegative and not s.is_positive)
        return SignFacts(positive=frozenset(s.name for s in syms if s.is_positive),
                         nonnegative=nonnegative | unsigned,
                         integer=frozenset(s.name for s in syms if s.is_integer) | unsigned)

    def assumptions(self, symbols: Iterable[sympy.Symbol]) -> list[sympy.logic.boolalg.Boolean]:
        facts = []
        for sym in symbols:
            if sym.is_integer or sym.name in self.integer:
                facts.append(sympy.Q.integer(sym))
            if sym.name in self.positive:
                facts.append(sympy.Q.positive(sym))
            elif sym.name in self.nonnegative:
                facts.append(sympy.Q.nonnegative(sym))
        return facts


def ordered_range(begin: symbolic.SymbolicType, end: symbolic.SymbolicType,
                  step: symbolic.SymbolicType | None) -> Bounds | None:
    """Inclusive ``(low, high)`` of ``begin..end`` by ``step``, or None when the sign of ``step`` is unknown: guessing
    ascending would prove a negative exponent non-negative."""
    if step is None:
        return None
    step = sympy.sympify(step)  # a range step may be a raw Python int, which has no is_positive
    if step.is_positive:
        return (begin, end)
    if step.is_negative:
        return (end, begin)
    return None


def loop_range(loop: LoopRegion) -> Bounds | None:
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    if start is None or end is None:
        return None
    return ordered_range(start, end, loop_analysis.get_loop_stride(loop))


def proven_nonnegative(exp: sympy.Expr, ranges: Ranges, facts: SignFacts) -> bool:
    """Whether ``exp`` is ``>= 0`` at the minimizing corner of the iterator ranges it is affine in."""
    corners = {}
    for sym in exp.free_symbols:
        if sym.name not in ranges:
            continue
        coeff = sympy.diff(exp, sym)
        if not coeff.is_number:
            return False  # non-affine in an iterator: no simple corner minimum
        low, high = ranges[sym.name]
        corners[sym] = high if coeff.is_negative else low
    residual = equalize_symbol(exp.subs(corners) if corners else exp)
    with sympy.assuming(*facts.assumptions(residual.free_symbols)):
        return sympy.ask(sympy.Q.nonnegative(residual)) is True


def relaxed_exponent(exp: sympy.Expr, ranges: Ranges, facts: SignFacts) -> sympy.Expr | None:
    """The integer exponent to feed ``ipow``, or None to keep ``pow``."""
    if exp.is_Number:
        if exp.is_integer:
            value = int(exp)
        elif exp.is_real and float(exp) == int(float(exp)):
            value = int(float(exp))  # integer-valued float literal (2.0 -> 2)
        else:
            return None  # genuinely fractional (0.5 -> sqrt)
        return sympy.Integer(value) if value >= 0 else None  # negative: a reciprocal
    with sympy.assuming(*facts.assumptions(exp.free_symbols)):
        if sympy.ask(sympy.Q.integer(exp)) is not True:
            return None
    return exp if proven_nonnegative(exp, ranges, facts) else None


@dataclass(slots=True)
class PowerRelaxer:
    """One walk over an SDFG tree that rewrites every provable ``Pow`` in sizes, subscripts, bounds, conditions,
    interstate assignments and symbol mappings."""
    relaxed: int = 0

    def relax(self, expr: Any, ranges: Ranges, facts: SignFacts) -> Any:
        core = expr.expr if isinstance(expr, symbolic.SymExpr) else expr
        if not isinstance(core, sympy.Basic) or not core.has(sympy.Pow):
            return expr

        def to_ipow(base: sympy.Expr, exp: sympy.Expr) -> sympy.Expr:
            result = relaxed_exponent(exp, ranges, facts)
            if result is None:
                return base**exp
            self.relaxed += 1
            return ipow(base, result if not result.free_symbols else exp)

        return core.replace(sympy.Pow, to_ipow)

    def relax_subset(self, sub: subsets.Subset, ranges: Ranges, facts: SignFacts) -> None:
        if isinstance(sub, subsets.Range):
            sub.ranges = [tuple(self.relax(component, ranges, facts) for component in rng) for rng in sub.ranges]
        elif isinstance(sub, subsets.Indices):
            sub.indices = [self.relax(idx, ranges, facts) for idx in sub.indices]

    def relax_descriptor(self, desc: data.Array, ranges: Ranges, facts: SignFacts) -> None:
        desc.shape = tuple(self.relax(item, ranges, facts) for item in desc.shape)
        desc.strides = tuple(self.relax(item, ranges, facts) for item in desc.strides)
        desc.offset = tuple(self.relax(item, ranges, facts) for item in desc.offset)
        desc.total_size = self.relax(desc.total_size, ranges, facts)

    def relax_text(self, text: str, ranges: Ranges, facts: SignFacts) -> str | None:
        """The rewritten Python expression, or None if it is unparseable or unchanged."""
        if not text or '**' not in text:
            return None
        try:
            expr = symbolic.pystr_to_symbolic(text)
        except Exception:  # pylint: disable=broad-exception-caught  # a non-symbolic statement is left as-is
            return None
        if not isinstance(expr, sympy.Basic) or not expr.has(sympy.Pow):
            return None
        relaxed = self.relax(expr, ranges, facts)
        if relaxed is expr:
            return None
        out = str(relaxed)
        return out if out != text else None

    def relax_code(self, code: CodeBlock | None, ranges: Ranges, facts: SignFacts) -> None:
        # Loop bounds and conditions codegen through the interstate-edge unparser, where an unrelaxed ``R**e``
        # becomes a ``double`` bound that can round to an extra iteration.
        if code is None:
            return
        relaxed = self.relax_text(code.as_string, ranges, facts)
        if relaxed is not None:
            code.as_string = relaxed

    def relax_assignments(self, assignments: dict[str, str], ranges: Ranges, facts: SignFacts) -> None:
        for var, value in list(assignments.items()):
            if isinstance(value, str):
                relaxed = self.relax_text(value, ranges, facts)
                if relaxed is not None:
                    assignments[var] = relaxed

    def relax_symbol_mapping(self, nsdfg: nodes.NestedSDFG, ranges: Ranges, facts: SignFacts) -> None:
        for name, value in list(nsdfg.symbol_mapping.items()):
            core = value.expr if isinstance(value, symbolic.SymExpr) else value
            if not isinstance(core, sympy.Basic) or not core.has(sympy.Pow):
                continue
            relaxed = self.relax(core, ranges, facts)
            if relaxed is not core:
                nsdfg.symbol_mapping[name] = relaxed

    def visit_sdfg(self, sdfg: SDFG, ranges: Ranges) -> None:
        self.visit_region(sdfg, ranges, SignFacts.of_sdfg(sdfg), set())

    def visit_region(self, region: ControlFlowRegion, ranges: Ranges, facts: SignFacts,
                     relaxed_arrays: set[str]) -> None:
        for iedge in region.edges():
            if iedge.data is None:
                continue
            self.relax_assignments(iedge.data.assignments, ranges, facts)
            self.relax_code(iedge.data.condition, ranges, facts)
        for block in region.nodes():
            if isinstance(block, LoopRegion):
                inner = dict(ranges)
                var = block.loop_variable
                if var:
                    rng = loop_range(block)
                    if rng is not None:
                        inner[str(var)] = rng
                    else:
                        inner.pop(str(var), None)  # rebound to an unknown range
                # The condition and the init see the iterator outside its body range (the condition fails at
                # ``i = end + step``; init runs before binding it), so they are relaxed under the enclosing ranges.
                self.relax_code(block.loop_condition, ranges, facts)
                self.relax_code(block.init_statement, ranges, facts)
                self.relax_code(block.update_statement, inner, facts)
                self.visit_region(block, inner, facts, relaxed_arrays)
            elif isinstance(block, SDFGState):
                self.visit_state(block, ranges, facts, relaxed_arrays)
            elif isinstance(block, ConditionalBlock):
                for condition, branch in block.branches:
                    self.relax_code(condition, ranges, facts)
                    self.visit_region(branch, ranges, facts, relaxed_arrays)
            elif isinstance(block, ControlFlowRegion):
                self.visit_region(block, ranges, facts, relaxed_arrays)

    def visit_state(self, state: SDFGState, ranges: Ranges, facts: SignFacts, relaxed_arrays: set[str]) -> None:
        sdfg = state.sdfg
        children = state.scope_children()
        scope_ranges: dict[nodes.EntryNode | None, Ranges] = {}

        def descend(entry: nodes.EntryNode | None, live: Ranges) -> None:
            scope_ranges[entry] = live
            for node in children[entry]:
                if isinstance(node, nodes.MapEntry):
                    self.relax_subset(node.map.range, live, facts)
                    inner = dict(live)
                    for conn in node.in_connectors:
                        if not conn.startswith('IN_'):
                            inner.pop(conn, None)
                    for param, (begin, end, step) in zip(node.map.params, node.map.range.ranges, strict=True):
                        bounds = ordered_range(begin, end, step)
                        if bounds is not None:
                            inner[str(param)] = bounds
                        else:
                            inner.pop(str(param), None)  # unknown-sign step: direction unknown
                    descend(node, inner)
                elif isinstance(node, nodes.NestedSDFG):
                    self.relax_symbol_mapping(node, live, facts)
                    self.visit_sdfg(node.sdfg, nested_ranges(node, live))
                elif isinstance(node, nodes.AccessNode) and node.data not in relaxed_arrays:
                    relaxed_arrays.add(node.data)
                    desc = sdfg.arrays.get(node.data)
                    if isinstance(desc, data.Array):
                        self.relax_descriptor(desc, live, facts)

        descend(None, ranges)

        scope = state.scope_dict()
        for edge in state.edges():
            if edge.data is None:
                continue
            live = scope_ranges.get(scope.get(edge.dst), ranges)
            for sub in (edge.data.subset, edge.data.other_subset):
                if sub is not None:
                    self.relax_subset(sub, live, facts)


def nested_ranges(nsdfg: nodes.NestedSDFG, ranges: Ranges) -> Ranges:
    """The outer ranges a nested SDFG sees through symbols mapped one-to-one onto an outer iterator."""
    inner: Ranges = {}
    for name, mapped in nsdfg.symbol_mapping.items():
        outer = symbolic.pystr_to_symbolic(mapped) if isinstance(mapped, str) else mapped
        if isinstance(outer, sympy.Symbol) and outer.name in ranges:
            inner[str(name)] = ranges[outer.name]
    return inner


@transformation.explicit_cf_compatible
class RelaxIntegerPowers(ppl.Pass):
    """Lower non-negative-integer ``Pow`` to ``ipow`` across the SDFG's size, subscript and bound expressions."""

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> set:
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """:return: The number of powers relaxed, or None if none was."""
        relaxer = PowerRelaxer()
        relaxer.visit_sdfg(sdfg, {})
        return relaxer.relaxed or None
