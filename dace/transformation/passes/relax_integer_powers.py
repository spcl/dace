# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``RelaxIntegerPowers`` -- lower ``base ** exp`` to ``ipow`` where the exponent
is a non-negative integer.

DaCe's symbolic C++ printer emits ``dace::math::pow`` (libm, ``double``) for a
non-constant exponent, which is illegal where an integer is required -- an array
size, a subscript, or a loop bound.

``Pow(base, exp) -> ipow(base, exp)`` whenever ``exp`` is a provable non-negative
integer: a non-negative integer constant, an integer-valued float literal, or a
symbolic integer proven ``>= 0`` by interval analysis over the enclosing iterator
ranges (``K - i - 1`` with ``for i in range(K)``.
"""
from typing import Any, Dict, Optional, Tuple

import sympy

from dace import SDFG, data, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.symbolic import equalize_symbol, ipow
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.analysis import loop_analysis

#: A live iteration range ``symbol name -> (low, high)`` (inclusive).
_Ranges = Dict[str, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]


def _ordered_range(
        begin: symbolic.SymbolicType, end: symbolic.SymbolicType,
        step: Optional[symbolic.SymbolicType]) -> Optional[Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]:
    """Inclusive ``(low, high)`` for an iterator ``begin..end`` stepping by ``step``.

    Direction needs the *provable* sign of ``step``. Unknown sign (``0:K:s``) -> which end is
    smaller is unknown -> ``None`` (no trusted range). Guessing ascending would "prove" a
    negative exponent non-negative and relax it to an out-of-range ``ipow``.
    """
    if step is None:
        return None
    step = sympy.sympify(step)  # a range step may be a raw Python int, which has no is_positive
    if step.is_positive:
        return (begin, end)
    if step.is_negative:
        return (end, begin)
    return None


def _loop_range(loop: LoopRegion) -> Optional[Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]:
    """Loop iterator's inclusive ``(low, high)``, or ``None`` if bounds or stride sign unknown."""
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    if start is None or end is None:
        return None
    return _ordered_range(start, end, loop_analysis.get_loop_stride(loop))


@transformation.explicit_cf_compatible
class RelaxIntegerPowers(ppl.Pass):
    """Lower non-negative-integer ``Pow`` to ``ipow`` across the SDFG's size,
    subscript and bound expressions."""

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Nodes

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        self._relaxed = 0
        self._pos = self._nonneg = self._int = frozenset()
        self._visit_sdfg(sdfg, {})
        return self._relaxed or None

    def _assumptions(self, symbols):
        """SymPy predicates for ``symbols`` under the current SDFG's declared
        signs / integrality (proven even when a duplicate symbol lost them)."""
        facts = []
        for sym in symbols:
            if sym.is_integer or sym.name in self._int:
                facts.append(sympy.Q.integer(sym))
            if sym.name in self._pos:
                facts.append(sympy.Q.positive(sym))
            elif sym.name in self._nonneg:
                facts.append(sympy.Q.nonnegative(sym))
        return facts

    def _proven_nonnegative(self, exp: sympy.Expr, ranges: Dict[str, Tuple[symbolic.SymbolicType,
                                                                           symbolic.SymbolicType]]) -> bool:
        """Is ``exp`` provably ``>= 0``? Ask SymPy under the declared assumptions."""
        corners = {}
        for sym in exp.free_symbols:
            if sym.name not in ranges:
                continue
            coeff = sympy.diff(exp, sym)
            if not coeff.is_number:
                return False  # non-affine in an iterator -> no simple corner minimum
            low, high = ranges[sym.name]
            corners[sym] = high if coeff.is_negative else low
        residual = equalize_symbol(exp.subs(corners) if corners else exp)
        with sympy.assuming(*self._assumptions(residual.free_symbols)):
            return sympy.ask(sympy.Q.nonnegative(residual)) is True

    def _relaxed_exponent(
            self, exp: sympy.Expr, ranges: Dict[str, Tuple[symbolic.SymbolicType,
                                                           symbolic.SymbolicType]]) -> Optional[sympy.Expr]:
        """The integer exponent to feed ``ipow``, or ``None`` to keep ``pow``."""
        if exp.is_Number:
            if exp.is_integer:
                value = int(exp)
            elif exp.is_real and float(exp) == int(float(exp)):
                value = int(float(exp))  # integer-valued float literal (2.0 -> 2)
            else:
                return None  # genuinely fractional (0.5 -> sqrt)
            return sympy.Integer(value) if value >= 0 else None  # negative -> reciprocal
        with sympy.assuming(*self._assumptions(exp.free_symbols)):
            if sympy.ask(sympy.Q.integer(exp)) is not True:
                return None
        return exp if self._proven_nonnegative(exp, ranges) else None

    def _relax(self, expr, ranges: Dict[str, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]):
        """Rewrite each provable ``Pow`` in ``expr`` to ``ipow``."""
        core = expr.expr if isinstance(expr, symbolic.SymExpr) else expr
        if not isinstance(core, sympy.Basic) or not core.has(sympy.Pow):
            return expr

        def to_ipow(base, exp):
            result = self._relaxed_exponent(exp, ranges)
            if result is None:
                return base**exp
            self._relaxed += 1
            return ipow(base, result if not result.free_symbols else exp)

        return core.replace(sympy.Pow, to_ipow)

    def _relax_subset(self, sub, ranges: Dict[str, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]) -> None:
        if isinstance(sub, subsets.Range):
            sub.ranges = [tuple(self._relax(component, ranges) for component in rng) for rng in sub.ranges]
        elif isinstance(sub, subsets.Indices):
            sub.indices = [self._relax(idx, ranges) for idx in sub.indices]

    def _relax_descriptor(self, desc: data.Array, ranges: Dict[str, Tuple[symbolic.SymbolicType,
                                                                          symbolic.SymbolicType]]) -> None:
        desc.shape = tuple(self._relax(item, ranges) for item in desc.shape)
        desc.strides = tuple(self._relax(item, ranges) for item in desc.strides)
        desc.offset = tuple(self._relax(item, ranges) for item in desc.offset)
        desc.total_size = self._relax(desc.total_size, ranges)

    def _relax_text(self, text: str, ranges: _Ranges) -> Optional[str]:
        """Relax provable ``Pow`` in a Python-expression string; return the rewritten
        text, or ``None`` if unparseable or unchanged. Powers carry Python ``**``, so a
        string without ``**`` needs no work."""
        if not text or '**' not in text:
            return None
        try:
            expr = symbolic.pystr_to_symbolic(text)
        except Exception:  # noqa: BLE001 -- a non-symbolic statement (e.g. a call) is left as-is
            return None
        if not isinstance(expr, sympy.Basic) or not expr.has(sympy.Pow):
            return None
        relaxed = self._relax(expr, ranges)
        if relaxed is expr:
            return None
        out = str(relaxed)
        return out if out != text else None

    def _relax_code(self, code, ranges: _Ranges) -> None:
        """Relax a :class:`~dace.properties.CodeBlock` (a loop bound / condition or a
        branch predicate) in place. These codegen through the interstate-edge unparser,
        NOT the descriptor path -- an un-relaxed ``R**e`` there becomes a ``dace::math::pow``
        (``double``) loop bound that can round to an extra iteration."""
        if code is None:
            return
        relaxed = self._relax_text(code.as_string, ranges)
        if relaxed is not None:
            code.as_string = relaxed

    def _relax_assignments(self, assignments: Dict[str, str], ranges: _Ranges) -> None:
        """Relax the RHS of each interstate-edge assignment in place."""
        for var, value in list(assignments.items()):
            if isinstance(value, str):
                relaxed = self._relax_text(value, ranges)
                if relaxed is not None:
                    assignments[var] = relaxed

    def _relax_symbol_mapping(self, nsdfg: nodes.NestedSDFG, ranges: _Ranges) -> None:
        """Relax each nested-SDFG symbol-mapping value (an outer-scope expression) in place."""
        for name, value in list(nsdfg.symbol_mapping.items()):
            core = value.expr if isinstance(value, symbolic.SymExpr) else value
            if not isinstance(core, sympy.Basic) or not core.has(sympy.Pow):
                continue
            relaxed = self._relax(core, ranges)
            if relaxed is not core:
                nsdfg.symbol_mapping[name] = relaxed

    def _nested_ranges(
        self, nsdfg: nodes.NestedSDFG, ranges: Dict[str, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]
    ) -> Dict[str, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]:
        """Carry outer ranges through a nested SDFG's symbol mapping."""
        inner: Dict[str, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]] = {}
        for name, value in nsdfg.symbol_mapping.items():
            value = symbolic.pystr_to_symbolic(value) if isinstance(value, str) else value
            if isinstance(value, sympy.Symbol) and value.name in ranges:
                inner[str(name)] = ranges[value.name]
        return inner

    def _visit_sdfg(self, sdfg: SDFG, ranges: Dict[str, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]) -> None:
        # ``sdfg.free_symbols`` yields names; the sign / integrality assumptions
        # live on the symbol objects in the array descriptors, so collect those.
        saved = (self._pos, self._nonneg, self._int)
        syms = set()
        for desc in sdfg.arrays.values():
            if isinstance(desc, data.Array):
                syms |= desc.free_symbols
        self._pos = frozenset(s.name for s in syms if s.is_positive)
        self._nonneg = frozenset(s.name for s in syms if s.is_nonnegative and not s.is_positive)
        self._int = frozenset(s.name for s in syms if s.is_integer)
        self._visit_region(sdfg, ranges, set())
        self._pos, self._nonneg, self._int = saved

    def _visit_region(self, region, ranges: Dict[str, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]],
                      relaxed_arrays: set) -> None:
        # Interstate edges carry symbol assignments + branch conditions that codegen
        # through the interstate-edge unparser (``x = R**k`` -> ``dace::math::pow``).
        for iedge in region.edges():
            if iedge.data is None:
                continue
            self._relax_assignments(iedge.data.assignments, ranges)
            self._relax_code(iedge.data.condition, ranges)
        for block in region.nodes():
            if isinstance(block, LoopRegion):
                inner = dict(ranges)
                var = block.loop_variable
                if var:
                    rng = _loop_range(block)
                    if rng is not None:
                        inner[str(var)] = rng
                    else:
                        inner.pop(str(var), None)  # rebound to an unknown range
                # condition + init see the iterator OUTSIDE its body range (condition fails at
                # ``i = end + step``; init runs pre-bind) -> relax under enclosing ranges, no own
                # iterator. update runs with in-body values -> keep them.
                self._relax_code(block.loop_condition, ranges)
                self._relax_code(block.init_statement, ranges)
                self._relax_code(block.update_statement, inner)
                self._visit_region(block, inner, relaxed_arrays)
            elif isinstance(block, SDFGState):
                self._visit_state(block, ranges, relaxed_arrays)
            elif isinstance(block, ConditionalBlock):
                for condition, branch in block.branches:
                    self._relax_code(condition, ranges)
                    self._visit_region(branch, ranges, relaxed_arrays)
            elif isinstance(block, ControlFlowRegion):
                self._visit_region(block, ranges, relaxed_arrays)

    def _visit_state(self, state: SDFGState, ranges: Dict[str, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]],
                     relaxed_arrays: set) -> None:
        sdfg = state.sdfg
        children = state.scope_children()
        scope_ranges = {}  # scope entry (or None) -> live ranges there

        def descend(entry, live: Dict[str, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]) -> None:
            scope_ranges[entry] = live
            for node in children[entry]:
                if isinstance(node, nodes.MapEntry):
                    self._relax_subset(node.map.range, live)
                    inner = dict(live)
                    for conn in node.in_connectors:
                        if not conn.startswith('IN_'):
                            inner.pop(conn, None)
                    for param, rng in zip(node.map.params, node.map.range.ranges):
                        prng = _ordered_range(rng[0], rng[1], rng[2])  # (begin, end, step)
                        if prng is not None:
                            inner[str(param)] = prng
                        else:
                            inner.pop(str(param), None)  # unknown-sign step: direction unknown
                    descend(node, inner)
                elif isinstance(node, nodes.NestedSDFG):
                    self._relax_symbol_mapping(node, live)
                    self._visit_sdfg(node.sdfg, self._nested_ranges(node, live))
                elif isinstance(node, nodes.AccessNode) and node.data not in relaxed_arrays:
                    relaxed_arrays.add(node.data)
                    desc = sdfg.arrays.get(node.data)
                    if isinstance(desc, data.Array):
                        self._relax_descriptor(desc, live)

        descend(None, ranges)

        scope = state.scope_dict()
        for edge in state.edges():
            if edge.data is None:
                continue
            live = scope_ranges.get(scope.get(edge.dst), ranges)
            for sub in (edge.data.subset, edge.data.other_subset):
                if sub is not None:
                    self._relax_subset(sub, live)
