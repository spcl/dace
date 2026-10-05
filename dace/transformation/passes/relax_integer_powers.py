# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``RelaxIntegerPowers`` -- lower ``base ** exp`` to ``ipow`` where the exponent is a non-negative integer.

DaCe's symbolic C++ printer emits ``dace::math::pow`` (libm, ``double``) for a non-constant exponent, which is
illegal where an integer is required -- an array size, a subscript, or a loop bound.

``Pow(base, exp) -> ipow(base, exp)`` whenever ``exp`` is a provable non-negative integer: a non-negative integer
constant, an integer-valued float literal, or a symbolic integer proven ``>= 0`` under the facts at the expression:
the SDFG's facts and the enclosing iterator ranges (``K - i - 1`` with ``for i in range(K)``).
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import sympy

from dace import SDFG, data, subsets, symbolic, symbolic_facts
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState, SymbolResolver
from dace.symbolic import ipow
from dace.transformation import pass_pipeline as ppl, transformation

#: The facts at the expression being relaxed, derived only when it holds a power.
FactsOf = Callable[[], symbolic.Facts]


def relaxed_exponent(exp: sympy.Expr, facts: symbolic.Facts) -> sympy.Expr | None:
    """The integer exponent to feed ``ipow``, or None to keep ``pow``."""
    if exp.is_Number:
        if exp.is_integer:
            value = int(exp)
        elif exp.is_real and float(exp) == int(float(exp)):
            value = int(float(exp))  # integer-valued float literal (2.0 -> 2)
        else:
            return None  # genuinely fractional (0.5 -> sqrt)
        return sympy.Integer(value) if value >= 0 else None  # negative: a reciprocal
    integer = symbolic_facts.with_integers(exp, facts.integers).is_integer
    return exp if integer and symbolic.provably_nonnegative(exp, facts) else None


@dataclass(slots=True)
class PowerRelaxer:
    """One walk over an SDFG tree that rewrites every provable ``Pow`` in sizes, subscripts, bounds, conditions,
    interstate assignments and symbol mappings."""
    resolver: SymbolResolver
    relaxed: int = 0

    def relax(self, expr: Any, facts_of: FactsOf) -> Any:
        core = expr.expr if isinstance(expr, symbolic.SymExpr) else expr
        if not isinstance(core, sympy.Basic) or not core.has(sympy.Pow):
            return expr
        facts = facts_of()

        def to_ipow(base: sympy.Expr, exp: sympy.Expr) -> sympy.Expr:
            result = relaxed_exponent(exp, facts)
            if result is None:
                return base**exp
            self.relaxed += 1
            return cast(sympy.Expr, ipow(base, result if not result.free_symbols else exp))

        return core.replace(sympy.Pow, to_ipow)

    def relax_subset(self, sub: subsets.Subset, facts_of: FactsOf) -> None:
        if isinstance(sub, subsets.Range):
            sub.ranges = [tuple(self.relax(component, facts_of) for component in rng) for rng in sub.ranges]
        elif isinstance(sub, subsets.Indices):
            # ``Indices`` sets ``indices`` in its constructor without declaring it
            sub.indices = [self.relax(idx, facts_of)
                           for idx in sub.indices]  # pyright: ignore[reportAttributeAccessIssue]

    def relax_descriptor(self, desc: data.Array, facts_of: FactsOf) -> None:
        desc.shape = tuple(self.relax(item, facts_of) for item in desc.shape)
        desc.strides = tuple(self.relax(item, facts_of) for item in desc.strides)
        desc.offset = tuple(self.relax(item, facts_of) for item in desc.offset)
        desc.total_size = self.relax(desc.total_size, facts_of)

    def relax_text(self, text: str, facts_of: FactsOf) -> str | None:
        """The rewritten Python expression, or None if it is unparseable or unchanged."""
        if not text or '**' not in text:
            return None
        try:
            expr = symbolic.pystr_to_symbolic(text)
        except Exception:  # pylint: disable=broad-exception-caught  # a non-symbolic statement is left as-is
            return None
        if not isinstance(expr, sympy.Basic) or not expr.has(sympy.Pow):
            return None
        relaxed = self.relax(expr, facts_of)
        if relaxed is expr:
            return None
        out = str(relaxed)
        return out if out != text else None

    def relax_code(self, code: CodeBlock | None, facts_of: FactsOf) -> None:
        # Loop bounds and conditions codegen through the interstate-edge unparser, where an unrelaxed ``R**e``
        # becomes a ``double`` bound that can round to an extra iteration.
        if code is None:
            return
        relaxed = self.relax_text(code.as_string, facts_of)
        if relaxed is not None:
            code.as_string = relaxed

    def relax_assignments(self, assignments: dict[str, str], facts_of: FactsOf) -> None:
        for var, value in list(assignments.items()):
            if isinstance(value, str):
                relaxed = self.relax_text(value, facts_of)
                if relaxed is not None:
                    assignments[var] = relaxed

    def relax_symbol_mapping(self, nsdfg: nodes.NestedSDFG, facts_of: FactsOf) -> None:
        for name, value in list(nsdfg.symbol_mapping.items()):
            core = value.expr if isinstance(value, symbolic.SymExpr) else value
            if not isinstance(core, sympy.Basic) or not core.has(sympy.Pow):
                continue
            relaxed = self.relax(core, facts_of)
            if relaxed is not core:
                nsdfg.symbol_mapping[name] = relaxed

    def visit_region(self, region: ControlFlowRegion, relaxed_arrays: set[str]) -> None:

        def inside() -> symbolic.Facts:
            return self.resolver.facts_at(region)

        for iedge in region.edges():
            if iedge.data is None:
                continue
            self.relax_assignments(iedge.data.assignments, inside)
            self.relax_code(iedge.data.condition, inside)
        for block in region.nodes():
            if isinstance(block, LoopRegion):
                # The condition and the init see the iterator outside its body range (the condition fails at
                # ``i = end + step``; init runs before binding it), so they are relaxed under the enclosing facts.
                self.relax_code(block.loop_condition, inside)
                self.relax_code(block.init_statement, inside)
                self.relax_code(block.update_statement, lambda loop=block: self.resolver.facts_at(loop))
                self.visit_region(block, relaxed_arrays)
            elif isinstance(block, SDFGState):
                self.visit_state(block, relaxed_arrays)
            elif isinstance(block, ConditionalBlock):
                for condition, branch in block.branches:
                    self.relax_code(condition, inside)
                    self.visit_region(branch, relaxed_arrays)
            elif isinstance(block, ControlFlowRegion):
                self.visit_region(block, relaxed_arrays)

    def visit_state(self, state: SDFGState, relaxed_arrays: set[str]) -> None:
        sdfg = state.sdfg
        children = state.scope_children()

        def at(node: nodes.Node) -> FactsOf:
            return lambda: self.resolver.facts_at(state, node)

        def descend(entry: nodes.EntryNode | None) -> None:
            for node in children[entry]:
                if isinstance(node, nodes.MapEntry):
                    self.relax_subset(node.map.range, at(node))
                    descend(node)
                elif isinstance(node, nodes.NestedSDFG):
                    self.relax_symbol_mapping(node, at(node))
                    self.visit_region(node.sdfg, set())
                elif isinstance(node, nodes.AccessNode) and node.data not in relaxed_arrays:
                    relaxed_arrays.add(node.data)
                    desc = sdfg.arrays.get(node.data)
                    if isinstance(desc, data.Array):
                        self.relax_descriptor(desc, at(node))

        descend(None)
        for edge in state.edges():
            if edge.data is None:
                continue
            for sub in (edge.data.subset, edge.data.other_subset):
                if isinstance(sub, subsets.Subset):
                    self.relax_subset(sub, at(edge.dst))


@transformation.explicit_cf_compatible
class RelaxIntegerPowers(ppl.Pass):
    """Lower non-negative-integer ``Pow`` to ``ipow`` across the SDFG's size, subscript and bound expressions."""

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """:return: The number of powers relaxed, or None if none was."""
        relaxer = PowerRelaxer(SymbolResolver())
        relaxer.visit_region(sdfg, set())
        return relaxer.relaxed or None
